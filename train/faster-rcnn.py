import datetime
import os
import torch
import torch.optim as optim
import torchvision
import cv2
from sklearn.model_selection import KFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import Gi4eDataset

# log the training process
def log_training_process_to_file(log_file, fold, epoch, train_loss, test_loss, accuracy):
  if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))
  with open(log_file, 'a') as f:
    f.write('Fold: ' + str(fold) + ', Epoch: ' + str(epoch) + ', Train Loss: ' + str(train_loss) +
            ', Test IoU Score: ' + str(test_loss) + ', Accuracy: ' + str(accuracy) + '\n')


def evaluate(net, images, device=None):
  net.eval()
  with torch.no_grad():
    if device is not None:
      images = images.to(device)

    outputs = net(images)

    # print(outputs)

    scores = []

    for output in outputs:
      scores.append(output['scores'])

    # _, predicted = torch.max(scores, 1)
    # print('Predicted: ', predicted)
  return outputs


def main(dataset_path, k_folds=10, epochs = 10, is_show_sample_image=False, stop_after_one_fold=False):
  current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  # get device to train on
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load the data
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

  # gi4e path to the dataset
  gi4e_dataset = Gi4eDataset(dataset_path, transform=transform)

  # define the optimizer
  net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  num_classes = 3  # 3 classes: background, left eye, right eye (0, 1, 2)
  in_features = net.roi_heads.box_predictor.cls_score.in_features
  net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  net = net.to(device)
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  batch_size = 12

  if is_show_sample_image:
    # show first image
    img, target = gi4e_dataset.get_image(0)
    # draw the bounding boxes, each line should have different color
    for box in target['boxes']:
      print('box: ', box)
      x1, y1, x2, y2 = box
      # draw the (0,0) point
      cv2.circle(img, (0, 0), 5, (0, 255, 0), -1)
      # draw the bounding box
      cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
      # draw x1, y1 coordinates
      cv2.circle(img, (int(x1), int(y1)), 5, (255, 0, 0), -1)
      # draw x2, y2 coordinates
      cv2.circle(img, (int(x2), int(y2)), 5, (0, 0, 255), -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)

  for fold, (train_ids, test_ids) in enumerate(kfold.split(gi4e_dataset)):
    print('Fold: ', fold)
    print('train_ids: ', train_ids)
    print('test_ids: ', test_ids)

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = torch.utils.data.DataLoader(dataset=gi4e_dataset,
                                               sampler=train_subsampler, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=gi4e_dataset,
                                              sampler=test_subsampler, batch_size=batch_size, shuffle=False, num_workers=2)

    # train the network
    for epoch in range(epochs):
      net.train()
      train_loss = 0.0
      # train the network
      for inputs, targets in train_loader:
        inputs = inputs.to(device)
        labels = []
        for i in range(len(targets['boxes'])):
          label = {}
          label['boxes'] = targets['boxes'][i].to(device)
          label['labels'] = targets['labels'][i].to(device)
          labels.append(label)
        optimizer.zero_grad()

        loss_dict = net(inputs, labels)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

      net.eval()
      count = 0
      test_total_iou = 0.0
      test_total_score = 0.0
      # test the network on the test data
      with torch.no_grad():
        for inputs, targets in test_loader:
          inputs = inputs.to(device)
          labels = []
          for i in range(len(targets['boxes'])):
            label = {}
            label['boxes'] = targets['boxes'][i].to(device)
            label['labels'] = targets['labels'][i].to(device)
            labels.append(label)

          outputs = net(inputs, labels)

          # calculate the loss for the test data via IoU
          for output_index in range(len(outputs)):
            output = outputs[output_index]
            output_boxes = output['boxes']
            output_labels = output['labels']
            output_scores = output['scores']

            target_boxes = labels[output_index]['boxes']
            target_labels = labels[output_index]['labels']

            for box_index in range(len(output_boxes)):
              box = output_boxes[box_index]
              box_label = output_labels[box_index]
              box_score = output_scores.tolist()[box_index]

              target_label_index = target_labels.tolist().index(box_label)
              target_box = target_boxes[target_label_index]

              # calculate the collision box
              collision_box = [max(box[0], target_box[0]), max(box[1], target_box[1]),
                               min(box[2], target_box[2]), min(box[3], target_box[3])]
              # calculate the area of the collision box
              collision_area = (collision_box[2] - collision_box[0]) * (collision_box[3] - collision_box[1])
              # calculate the area of the union box
              union_area = (box[2] - box[0]) * (box[3] - box[1]) + (target_box[2] - target_box[0]) * \
                  (target_box[3] - target_box[1]) - collision_area
              # calculate the IoU
              iou = abs(collision_area / union_area)

              count += 1
              test_total_iou += iou.item()
              test_total_score += box_score

      test_avg_iou = test_total_iou / count
      test_avg_score = test_total_score / count
      print('Fold ' + str(fold) + ', Epoch: ' + str(epoch) + ', Train loss: ' + str(train_loss) +
            ', Test average iou: ' + str(test_avg_iou) + ', Test average score: ' + str(test_avg_score))

      # save the report to the log file
      log_file = './logs/faster_rcnn/' + current_date + '.log'
      log_training_process_to_file(log_file, fold, epoch, train_loss, test_avg_iou, test_avg_score)

      # print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
      # save the model
      model_path = './models/faster_rcnn/' + current_date + '/fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth'
      if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
      torch.save(net.state_dict(), model_path)

    #   # for testing, break after one epoch
    #   break

    if stop_after_one_fold:
      break

  print('Finished Training')


if __name__ == '__main__':
  # dataset_path = './dataset/FasterRCNN/'
  dataset_path = './datasets/faster-rcnn/gi4e/'
  print('run the main function with the dataset path with 10 folds and 10 epochs')
  main(dataset_path, 10, 10)
  print('run the main function with the dataset path with 10 folds and 100 epochs')
  main(dataset_path, 10, 100, stop_after_one_fold=True)
