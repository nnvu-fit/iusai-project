import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import matplotlib.pyplot as plt

from data_set import Gi4eDataset


def get_labels(labels, device=None):
    targets = []
    bbox = labels['boxes']
    category_id = labels['labels']
    print(len(bbox))
    for i in range(len(bbox[0])):
        target = {}
        xmin, ymin, width, height = [bbox[0][i], bbox[1][i], bbox[2][i], bbox[3][i]]
        xmax = xmin + width
        ymax = ymin + height
        target['boxes'] = torch.stack([torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32).to(device)])
        target['labels'] = torch.stack([torch.tensor(category_id[i], dtype=torch.int64).to(device)])
        targets.append(target)
    return targets

def main():
    # get device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the optimizer
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load the data
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((640, 640)),
        torchvision.transforms.ToTensor()
    ])

    # dataset_path = './dataset/FasterRCNN/'
    dataset_path = './datasets/FasterRCNN/data/'
    # coco dataset
    train_set = torchvision.datasets.CocoDetection(
        root=dataset_path+'train',
        annFile=dataset_path+'train/_annotations.coco.json',
        transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=16, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CocoDetection(
        root=dataset_path+'valid',
        annFile=dataset_path+'valid/_annotations.coco.json',
        transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=16, shuffle=False, num_workers=2)

    # gi4e path to the dataset
    gi4e_path = './datasets/FasterRCNN/gi4e/'
    gi4e_dataset = Gi4eDataset(gi4e_path, transform=transform)

    # split the dataset into train and test
    train_size = int(0.8 * len(gi4e_dataset))
    test_size = len(gi4e_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(gi4e_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # # show the first image and the labels
    # image, labels = gi4e_dataset[0]
    # image = image.permute(1, 2, 0)
    # image = image.numpy()
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # for box in labels['boxes']:
    #     x_min, y_min, x_max, y_max = box
    #     point1 = (int(x_min), int(y_min))
    #     point2 = (int(x_max), int(y_max))
    #     image = cv2.rectangle(image, point1, point2, (0, 255, 0), 2)

    # plt.imshow(image)
    # plt.show()
    

    # train the network
    for epoch in range(1):
        print('epoch: ', epoch)
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
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
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = get_labels(labels, device=device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % ( 100 * correct / total))


    print('Finished Training')

if __name__ == '__main__':
    main()
