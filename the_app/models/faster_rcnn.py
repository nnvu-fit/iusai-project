
import io
import torch
import torchvision.models.detection as detection
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import helper.image_helper as ih

def detect_eyes(image):
    """
    Detect eyes in the image using Faster RCNN model.
    :param image: Input image
    :return: Image with detected eyes highlighted
    """
    # get device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the data
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the pre-trained Faster RCNN model
    net = detection.fasterrcnn_resnet50_fpn(weights=None)
    # weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 3  # 3 classes: background, left eye, right eye (0, 1, 2)
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # read image
    image = Image.open(io.BytesIO(image))
    tensor_image = transform(image).unsqueeze(0)
    tensor_image = tensor_image.to(device)

    # Load the model weights
    net.load_state_dict(torch.load('./weights/faster-rcnn.pth', map_location=device, weights_only=False))
    net = net.to(device)
    net.eval()
    # Perform inference
    with torch.no_grad():
        predictions = net(tensor_image)

    # Move predictions to CPU for further processing
    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # create a composed image with from cropped eyes from the original image
    new_width = 0
    new_height = 0
    for box_index, box in enumerate(boxes):
        box_score = scores[box_index]
        if box_score < 0.5:
            continue
        label = labels[box_index]
        label_str = 'left' if label == 1 else 'right'
        eye = image.crop((box[0], box[1], box[2], box[3]))
        eye_width, eye_height = eye.size
        new_width += eye_width
        new_height += eye_height
    # calculate the maximum width and height of the new image
    # create a new image with the maximum width and height
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    # paste the cropped eyes into the new image
    x_offset = 0
    y_offset = 0
    for box_index, box in enumerate(boxes):
        box_score = scores[box_index]
        if box_score < 0.5:
            continue
        label = labels[box_index]
        label_str = 'left' if label == 1 else 'right'
        eye = image.crop((box[0], box[1], box[2], box[3]))
        eye_width, eye_height = eye.size
        new_image.paste(eye, (x_offset, y_offset))
        x_offset += eye_width
        y_offset += eye_height
        if x_offset + eye_width > new_width:
            x_offset = 0
            y_offset += eye_height
        if y_offset + eye_height > new_height:
            y_offset = 0
            x_offset += eye_width

    # draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    for box_index, box in enumerate(boxes):
        box_score = scores[box_index]
        if box_score < 0.5:
            continue
        label = labels[box_index]
        label_str = 'left' if label == 1 else 'right'
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, box[1]), label_str, fill="green")
        draw.text((box[0] + 4, box[1] + 10), str(box_score*100), fill="green")

    return image, new_image
