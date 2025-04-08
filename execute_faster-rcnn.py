import datetime
import os
import torch
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import Gi4eDataset


def main(dataset_path, weights_path, eyes_dataset_path, is_log_enabled=False):
  def log(*values):
    if is_log_enabled:
      print(values)

  current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  eyes_dataset_path = os.path.join(eyes_dataset_path, current_date)
  if not os.path.exists(eyes_dataset_path):
    os.makedirs(eyes_dataset_path)

  # get device to train on
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load the data
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

  # gi4e path to the dataset
  gi4e_dataset = Gi4eDataset(dataset_path, transform=transform)
  
  # define the optimizer
  net = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
  # weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  num_classes = 3  # 3 classes: background, left eye, right eye (0, 1, 2)
  in_features = net.roi_heads.box_predictor.cls_score.in_features
  net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  # load the model state
  net.load_state_dict(torch.load(weights_path, map_location=device))

  net = net.to(device)

  batch_size = 12
  dataloader = torch.utils.data.DataLoader(gi4e_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

  user_number_dict = {}

  net.eval()
  # extract eyes from the images and labels
  with torch.no_grad():
    for (images, targets) in dataloader:
      images = images.to(device)
      outputs = net(images)

      user_numbers = [value[0] for value in targets['user_number'].numpy()]
      log('user_numbers: ', user_numbers)
      for i, (output) in enumerate(outputs):
        log('user_numbers length: ', len(user_numbers))
        log('index: ', i)

        if user_numbers[i] not in user_number_dict.keys():
          user_number_dict[user_numbers[i]] = 0
        user_number_dict[user_numbers[i]] += 1

        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_prefix = str(user_numbers[i]) + '_' + str(user_number_dict[user_numbers[i]])
        # cv2.imshow('image', image)
        labels = output['labels']
        scrores = output['scores']
        log('scores: ', scrores)
        for box_index, box in enumerate(output['boxes']):
          box_score = scrores[box_index].item()
          if box_score < 0.5:
            continue
          label = labels[box_index].item()
          label_str = 'left' if label == 1 else 'right'
          # crop the left eye from the image
          eye = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

          # save the eye image to the file
          eye_path = os.path.join(eyes_dataset_path, str(user_numbers[i]))
          if not os.path.exists(eye_path):
            os.makedirs(eye_path)
          eye_path = os.path.join(eye_path, image_prefix + '_' + label_str + '.png')
          while (os.path.exists(eye_path)):
            eye_path = eye_path.replace('.png', '_1.png')
            
          log('eye_path: ', eye_path)
          try:
            if eye.shape[0] == 0 or eye.shape[1] == 0:
              log('eye shape: ', eye.shape)
              continue
            success = cv2.imwrite(eye_path, 255*eye)
            if not success:
              print('Error saving the image: ', eye_path)
              print('eye: ', eye)
          except Exception as e:
            print('Error saving the image: ', eye_path)
            print('eye: ', eye)
            print('Exception: ', e)
            cv2.imshow('eye', eye)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    


if __name__ == "__main__":
  dataset_path = './datasets/faster-rcnn/gi4e/'
  eyes_dataset_path = './datasets/faster-rcnn/gi4e_eyes/'
  weights_path = './models/faster_rcnn/20250117_161726/fold_4_epoch_9.pth'
  main(dataset_path, weights_path, eyes_dataset_path)