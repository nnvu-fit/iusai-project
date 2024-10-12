import os
import cv2

def extract_images_from_videos(video_list_location, image_list_location, inteval=None):
    # validate video_list_location if it is a valid path
    if not os.path.exists(video_list_location) and not os.path.isfile(video_list_location):
        print('Invalid video list location')
        return
    
    # create image_list_location if it does not exist
    if not os.path.exists(image_list_location):
        os.makedirs(image_list_location)

    video_list = []
    # get all video files belong to video_list_location
    if os.path.isfile(video_list_location):
        video_list.append(video_list_location)
    else:
        for root, dirs, files in os.walk(video_list_location):
            for file in files:
                video_list.append(os.path.join(root, file))

    for video_file in video_list:
        video_file = video_file.strip()
        # get video name from video file name
        video_name = video_file.split('/')[-1].split('\\')[-1].split('.')[0]
        # get image directory output path for the video
        image_path = os.path.join(image_list_location, video_name)
        print('Extracting images from ' + video_name + ' to ' + image_path)
        # does not process if image directory already exist
        if os.path.exists(image_path):
            print('Image directory already exist for ' + video_name)
            continue
        # create image directory
        os.makedirs(image_path)
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
            cv2.imwrite(image_path + '/' + 'frame' + str(frame_count) + '.jpg', image)
            # Read next frame
            success, image = cap.read()
            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    video_list_location = './datasets/FasterRCNN/video-data'
    image_list_location = './datasets/FasterRCNN/image-data'
    extract_images_from_videos(video_list_location, image_list_location)
    print('Extract images from videos successfully')