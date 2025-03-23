import glob
import os
import threading

import cv2
import numpy as np
import pandas as pd

from IPython.display import YouTubeVideo


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
        # check if cpu usage is less than 80%, then increase the max_threads by 1
        # otherwise, decrease the max_threads by a factor of 2
        if len(threads) < max_threads:
            max_threads = int(max_threads + 1)
        else:
            max_threads = int(max_threads / 2)
        # check if there are any threads that are stopped
        for thread in threads:
            if not thread.is_alive():
                threads.remove(thread)
        
        while len(threads) < max_threads:
            row = file_information.iloc[index]
            videoID = row['videoID']
            label = row['personName']
            file_path = row['file_path']

            print(index, '/', number_of_videos, '- Processing video: ',
                  videoID, ' with label: ', label)

            output_folder = os.path.join(output_path, label)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            threads.append(threading.Thread(
                target=extract_video_frames,
                args=(videoID, file_path, output_folder)))
            threads[-1].start()

            index += 1


def extract_video_frames(videoID, file_path, output_folder):
    video_file = np.load(file_path)

    colorImages = video_file['colorImages']
    for image_index in range(colorImages.shape[3]):
        image = colorImages[:, :, :, image_index]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = os.path.join(
            output_folder, videoID + '_frame_' + str(image_index) + '.jpg')
        if not os.path.exists(image_path):
            cv2.imwrite(image_path, image)


if __name__ == '__main__':
    input_path = '../data/YouTubeFacesWithFacialKeypoints'
    output_path = '../datasets/YouTubeFacesWithFacialKeypoints'

    main(input_path, output_path)
