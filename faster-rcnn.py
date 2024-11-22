import datetime
import os
import torch
import torch.optim as optim
import torchvision
import cv2
from sklearn.model_selection import KFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_set import Gi4eDataset
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def evaluate(net, images, device = None):
    net.eval()
    with torch.no_grad():
        if device is not None:
            images = images.to(device)

        outputs = net(images)

        #print(outputs)

        scores = []

        for output in outputs:
            scores.append(output['scores'])

        #_, predicted = torch.max(scores, 1)
        #print('Predicted: ', predicted)
    return outputs

def main(dataset_path):

    # get device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the optimizer
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            #weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 2
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load the data
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((640, 640)),
        torchvision.transforms.ToTensor()
    ])

    batch_size = 12
    k_folds = 5
    # gi4e path to the dataset
    gi4e_dataset = Gi4eDataset(dataset_path, transform=transform)

    # show first image
    img, target = gi4e_dataset.get_image(0)
    # draw the bounding boxes, each line should have different color
    for box in target['boxes']:
        print('box: ', box)
        x1, y1, x2, y2 = box
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
        # split the dataset into train and test
        #train_size = int(0.8 * len(gi4e_dataset))
        #test_size = len(gi4e_dataset) - train_size
        #train_dataset, test_dataset = torch.utils.data.random_split(
        #   gi4e_dataset, [train_size, test_size])

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
        for epoch in range(10):
            print('epoch: ', epoch)
            net.train()
            running_loss = 0.0
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
                running_loss += loss.item()
                # if i % 20 == 19:
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 20))
                #     running_loss = 0.0
            print('epoch loss: ', running_loss)

            net.eval()
            correct = 0
            total = 0
            count = 0
            with torch.no_grad():
                  for images, targets in test_loader:
                      if device is not None:
                          images = images.to(device)

                      predicted= net(images)
                      #predicted = evaluate(net, images, device)
                      count = count + 1
                      print('image: ', count, '=>', predicted)
                      test_labels = []
                      for i in range(len(targets['boxes'])):
                          label = {}
                          label['boxes'] = targets['boxes'][i].to(device)
                          label['labels'] = targets['labels'][i].to(device)
                          test_labels.append(label)

                      #total += test_labels.size(0)
                      #correct += (predicted == test_labels).sum().item()

            #print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
            #save the model
            model_path = './models/faster_rcnn/' + current_date + '/fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth'
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            torch.save(net.state_dict(), model_path)

    print('Finished Training')


if __name__ == '__main__':
    # dataset_path = './dataset/FasterRCNN/'
    dataset_path = './datasets//faster-rcnn/gi4e/'
    main(dataset_path)
