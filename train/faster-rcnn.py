import os
import sys

import datetime
import torch
import torch.optim as optim
import torchvision
import cv2
from sklearn.model_selection import KFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import trainer
import dataset as ds

k_folds = 10
epochs = 10


def log_training_process_to_file(log_file, fold, epoch, train_loss, test_loss, accuracy):
  """
  Logs the training process details to a specified file.

  Args:
    log_file (str): Path to the log file.
    fold (int): Current fold number in K-Fold cross-validation.
    epoch (int): Current epoch number.
    train_loss (float): Training loss for the current epoch.
    test_loss (float): Test IoU score for the current epoch.
    accuracy (float): Accuracy or average score for the current epoch.

  Returns:
    None
  """
  # create the log file if it does not exist
  if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))
  # create the log file
  with open(log_file, 'a') as f:
    f.write('Fold: ' + str(fold) + ', Epoch: ' + str(epoch) + ', Train Loss: ' + str(train_loss) +
            ', Test IoU Score: ' + str(test_loss) + ', Accuracy: ' + str(accuracy) + '\n')


def evaluate(net, images, device=None):
  """
  Evaluates the given model on a batch of images.

  Args:
    net (torch.nn.Module): The trained Faster R-CNN model.
    images (torch.Tensor): A batch of images to evaluate.
    device (torch.device, optional): The device to run the evaluation on (e.g., 'cuda' or 'cpu').

  Returns:
    list: A list of dictionaries containing the outputs for each image, including bounding boxes, labels, and scores.
  """
  # set the model to evaluation mode
  net.eval()
  with torch.no_grad():
    # move the images to the device if specified
    if device is not None:
      images = images.to(device)
    # get the outputs from the model
    outputs = net(images)
    # initialize an empty list to store the scores
    scores = []
    # iterate through the outputs and extract the scores
    for output in outputs:
      scores.append(output['scores'])
  # return the outputs
  return outputs


def main(dataset, is_show_sample_image=False, stop_after_one_fold=False):
  current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  # save the report to the log file
  log_file = './logs/faster_rcnn/' + current_date + '.log'

  # get device where the code will run
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # define the optimizer
  net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
      weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  # weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  num_classes = 3  # 3 classes: background, left eye, right eye (0, 1, 2)
  in_features = net.roi_heads.box_predictor.cls_score.in_features
  net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  net = net.to(device)
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  batch_size = 16

  if is_show_sample_image:
    # show first image
    img, target = dataset.get_image(0)
    # draw the bounding boxes, each line should have different color
    for box in target['boxes']:
      print('box: ', box)
      x1, y1, x2, y2 = box
      # draw the (0,0) point
      cv2.circle(img, (0, 0), 5, (0, 255, 0), -1)
      # draw the bounding box
      cv2.rectangle(img, (int(x1), int(y1)),
                    (int(x2), int(y2)), (0, 255, 0), 2)
      # draw x1, y1 coordinates
      cv2.circle(img, (int(x1), int(y1)), 5, (255, 0, 0), -1)
      # draw x2, y2 coordinates
      cv2.circle(img, (int(x2), int(y2)), 5, (0, 0, 255), -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)

  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print('Fold: ', fold, '/', str(k_folds))
    print('train_ids: ', train_ids)
    print('test_ids: ', test_ids)

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               sampler=train_subsampler,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               collate_fn=trainer.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              sampler=test_subsampler,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2,
                                              collate_fn=trainer.collate_fn)

    # train the network
    fold_start_time = datetime.datetime.now()
    for epoch in range(epochs):
      epoch_start_time = datetime.datetime.now()
      net.train()
      train_loss = 0.0
      # train the network
      for inputs, targets in train_loader:
        inputs = inputs.to(device)
        labels = []
        for target in targets:
          label = {}
          label['boxes'] = target['boxes'].to(device)
          label['labels'] = target['labels'].to(device)
          labels.append(label)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss_dict = net(inputs, labels)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      epoch_end_time = datetime.datetime.now()
      epoch_duration = epoch_end_time - epoch_start_time
      print('Fold ' + str(fold) + ', Epoch: ' + str(epoch) +
            ', Duration: ' + str(epoch_duration) + ', Train loss: ' + str(train_loss))
      net.eval()
      count = 0
      test_total_iou = 0.0
      test_total_score = 0.0
      # test the network on the test data
      with torch.no_grad():
        for inputs, targets in test_loader:
          inputs = inputs.to(device)
          labels = []
          for target in targets:
            label = {}
            label['boxes'] = target['boxes'].to(device)
            label['labels'] = target['labels'].to(device)
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
              collision_area = (
                  collision_box[2] - collision_box[0]) * (collision_box[3] - collision_box[1])
              # calculate the area of the union box
              union_area = (box[2] - box[0]) * (box[3] - box[1]) + (target_box[2] - target_box[0]) * \
                  (target_box[3] - target_box[1]) - collision_area
              # calculate the IoU
              iou = abs(collision_area / union_area)

              count += 1
              test_total_iou += iou.item()
              test_total_score += box_score

      # calculate the average IoU and score for the test data
      if count == 0:
        test_avg_iou = 0.0
        test_avg_score = 0.0
      else:
        test_avg_iou = test_total_iou / count
        test_avg_score = test_total_score / count
      print('Fold ' + str(fold) + ', Epoch: ' + str(epoch) + ', Train loss: ' + str(train_loss) +
            ', Test average iou: ' + str(test_avg_iou) + ', Test average score: ' + str(test_avg_score))

      log_training_process_to_file(
          log_file, fold, epoch, train_loss, test_avg_iou, test_avg_score)

      # print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
      # save the model
      model_path = './models/faster_rcnn/' + current_date + \
          '/fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth'
      if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
      torch.save(net.state_dict(), model_path)

    #   # for testing, break after one epoch
    #   break
    fold_end_time = datetime.datetime.now()
    fold_duration = fold_end_time - fold_start_time
    print('Fold ' + str(fold) + ' duration: ' + str(fold_duration))

    if stop_after_one_fold:
      break

  print('Finished Training')


if __name__ == '__main__':
  # set the transform for the dataset
  transform = torchvision.transforms.Compose(
      [
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.ToTensor()
      ])

  # train the network with the dataset path of gi4e
  # set the dataset path to the gi4e dataset
  # dataset_path = './dataset/FasterRCNN/'
  dataset_path = './datasets/GI4E/'
  # gi4e path to the dataset
  gi4e_dataset = ds.Gi4eDataset(dataset_path, transform=transform)
  print('Train the network with the gi4e dataset')
  print('run the main function with the dataset path with 10 folds and 10 epochs')
  main(gi4e_dataset)
  # print('run the main function with the dataset path with 10 folds and 100 epochs')
  # main(gi4e_dataset, stop_after_one_fold=True)

  # # train the network with the dataset path of YouTubeFacesWithFacialKeypoints
  # dataset_path = '.\datasets\YouTubeFacesWithFacialKeypoints'
  # # YouTubeFacesWithFacialKeypoints path to the dataset
  # youtube_faces_dataset = ds.YoutubeFacesWithFacialKeypoints(
  #   dataset_path,
  #   is_classification=False,
  #   transform=transform,
  #   number_of_samples=50)
  # print('Train the network with the youtube faces dataset')
  # print('run the main function with the dataset path with 10 folds and 10 epochs')
  # main(youtube_faces_dataset)
  # print('run the main function with the dataset path with 10 folds and 100 epochs')
  # main(youtube_faces_dataset, stop_after_one_fold=True)
