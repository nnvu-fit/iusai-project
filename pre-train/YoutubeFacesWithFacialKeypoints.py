import glob
import json
import os
import threading

import cv2
import numpy as np
import pandas as pd

import psutil
from time import sleep

# Extend the JSONEncoder class
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def main(input_path, output_path):
    #  get all the files in the input path
    files = glob.glob(input_path + '/**', recursive=True)
    #  get all the files with the extension .csv
    label_files = [f for f in files if f.endswith('.csv')]
    # get all the files with the extension .npz
    npz_files = [f for f in files if f.endswith('.npz')]
    npz_files = pd.DataFrame(npz_files, columns=['file_path'])
    npz_files['videoID'] = npz_files['file_path'].apply(
        lambda x: os.path.basename(x).split('.')[0])

    file_information = pd.DataFrame()
    # each file in the label_files list is the path to a csv file
    for file in label_files:
        # open the file in read mode
        labels = pd.read_csv(file)
        labels = labels.join(npz_files.set_index(
            'videoID'), on='videoID', how='inner')
        file_information = pd.concat([file_information, labels])

    print(file_information.head())

    number_of_videos = file_information['videoID'].nunique()
    print('Number of videos: ', number_of_videos)

    threads = []
    index = 0
    max_threads = 10
    while index < number_of_videos:
        print('------------------------------------')
        # check if cpu usage is less than 80%, then increase the max_threads by 1
        # otherwise, decrease the max_threads by a factor of 2
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        print('CPU Usage: ', cpu_usage, ' RAM Usage: ', ram_usage)
        if cpu_usage < 80 and ram_usage < 80:
            max_threads += 1
        else:
            max_threads = max_threads // 2

        # check if there are any threads that are stopped
        for thread in threads:
            if not thread.is_alive():
                threads.remove(thread)
        print('Max Threads: ', max_threads, ' Threads: ', len(threads))

        # check if the number of threads is less than the max_threads and the index is less than the number of videos
        # then start a new thread
        # the new thread will extract the frames from the video
        # and save them in the output folder
        while len(threads) < max_threads and index < number_of_videos:
            row = file_information.iloc[index]
            videoID = row['videoID']
            label = row['personName']
            file_path = row['file_path']

            print(index + 1, '/', number_of_videos, '- Processing video: ',
                  videoID, ' with label: ', label)

            output_folder = os.path.join(output_path, label)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            threads.append(threading.Thread(
                target=extract_video_frames,
                args=(videoID, label, file_path, output_folder)))
            threads[-1].start()

            index += 1

        sleep(0.5)


def extract_video_frames(videoID, label, file_path, output_folder):
    video = np.load(file_path)

    # get the color images from the video
    colorImages = video['colorImages']
    # get the bounding boxes, landmarks 2D and landmarks 3D from the video
    bounding_boxes = video['boundingBox']
    landmarks_2d = video['landmarks2D']
    landmarks_3d = video['landmarks3D']

    # define the annotations for the video
    annotations = []
    for image_index in range(colorImages.shape[3]):
        image = colorImages[:, :, :, image_index]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = os.path.join(
            output_folder, videoID + '_frame_' + str(image_index) + '.jpg')
        if not os.path.exists(image_path):
            cv2.imwrite(image_path, image)
        annotations.append({
            'label': label,
            'image_id': videoID + '_frame_' + str(image_index),
            'bounding_box': bounding_boxes[:, :, image_index],
            'landmarks_2d': landmarks_2d[:, :, image_index],
            'landmarks_3d': landmarks_3d[:, :, image_index]
        })

    # save the annotations in a json file
    annotations_path = os.path.join(
        output_folder, videoID + '_annotations.json')
    if not os.path.exists(annotations_path):
        with open(annotations_path, 'w') as f:
            annotation_json = json.dumps(annotations, cls=NumpyEncoder)
            f.write(annotation_json)


if __name__ == '__main__':
    input_path = './data/YouTubeFacesWithFacialKeypoints'
    output_path = 'C:/Users/nnvuf/Workspace/datasets/YouTubeFacesWithFacialKeypoints'

    main(input_path, output_path)
