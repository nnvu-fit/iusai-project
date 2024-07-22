import os
import cv2

# extract images from video location
# video_path: path to videos
# image_path: path to save images
# image_name: name of images followed by number + type of image + jpg
# folder structure:
#  gt.txt: ground truth file (start time\tend time\tlabel)
#  *.wav: audio file (ignored)
#  *1.avi: video file (captured by computer's camera)
#  *2.avi: video file (captured by test taker's camera) (ignored)
# Label detail:
#  0: no action
#  1: looking at the screen


def extract_images_from_videos(video_list_location, image_path, inteval=None):
    # validate video_list_location and image_path
    success, video_file, gt_file = validate_video_path(video_list_location)
    if not success:
        return
    image_prefix = video_list_location.split('/')[-1]
    # read gt.txt file and save to gt
    gt = read_gt_file(gt_file)
    # get image prefix from video file name
    label_dict = [['0', 0]]
    for x in gt:
        if [x[2], 0] not in label_dict:
            label_dict.append([x[2], 0])
    print('Label dict of ' + video_file + ' (start): ' + str(label_dict))
    # read video file and extract images within start time and end time to image_path follow the format: image_path/label/image_name.jpg
    # image_name: image_prefix + '_' + frame_count
    cap = cv2.VideoCapture(video_file)
    success, image = cap.read()
    frame_count = 0
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #calculate the time interval between frames
    time_interval = 1 if inteval == None else int(fps/inteval)
    while success:
        # Skip frames if this frame is not the frame we want
        if frame_count % time_interval != 0:
            frame_count += 1
            success, image = cap.read()
            continue
        # # Find label of current frame
        # frame_label_list = [l for [s, e, l] in gt if cap.get(cv2.CAP_PROP_POS_MSEC) >= float(
        #     s) * 1000 and cap.get(cv2.CAP_PROP_POS_MSEC) < float(e) * 1000]
        # # If no label found, set label to 0
        # if frame_label_list == []:
        #     frame_label_list.append('0')
        # Get the first label in the list
        frame_label = image_prefix
        # Create folder for label if not exist
        if not os.path.exists(image_path + frame_label):
            os.makedirs(image_path + frame_label)
        cv2.imwrite(image_path + frame_label + '/' + image_prefix +
                    '_' + str(frame_count) + '.jpg', image)
        # # Increase label count by 1
        # label_count = [l for l in label_dict if l[0] == frame_label][0]
        # label_count[-1] += 1
        # Read next frame
        success, image = cap.read()
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

    # Print label dict
    print('Label dict of ' + video_file + ' (end): ' + str(label_dict))
    return label_dict

def validate_video_path(video_path):
    # add '/' to the end of video_path if not exist
    if video_path[-1] != '/':
        video_path += '/'
    # list all files in video_path
    video_list = os.listdir(video_path)
    # find gt.txt file
    gt_file_found = [file for file in video_list if file.endswith('.txt')]
    gt_file = gt_file_found[0] if gt_file_found != [] else []
    # find video file captured by computer's camera
    video_file_found = [file for file in video_list if file.endswith('1.avi')]
    video_file = video_file_found[0] if video_file_found != [] else []
    # Return if no video file or gt.txt file found
    # print error message if no video file or gt.txt file found
    if video_file == [] or gt_file == []:
        print('No video file or gt.txt file found')
        return False, '', ''
    else:
        return True, video_path + video_file, video_path + gt_file

def read_gt_file(gt_file_path):
    # read gt.txt file and save to gt
    gt = []
    with open(gt_file_path, 'r') as f:
        for line in f:
            gt.append(line.strip().split('\t'))
    return gt