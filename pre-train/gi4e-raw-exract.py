import os
import sys

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def main(input_path, output_path):
  """
  Main function to extract and save the data from the input path to the output path.
  """

  import os
  import glob
  import cv2

  # Check if the input path exists
  if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input path {input_path} does not exist.")
  # Create the output directory if it does not exist
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  # Get all the files in the input path
  files = glob.glob(os.path.join(input_path, '**/*.png'), recursive=True)
  # Check if any files were found
  if not files:
    raise FileNotFoundError(f"No files found in the input path {input_path}.")
  
  # Get all the label files in the input path
  label_files = glob.glob(os.path.join(input_path, '**/*.txt'), recursive=True)
  label_files = [f for f in label_files if 'labels' in os.path.basename(f)]
  # Check if any label files were found
  if not label_files:
    raise FileNotFoundError(f"No label files found in the input path {input_path}.")
  
  # Loop through each label file and extract the corresponding image
  for label_file in label_files:
    # Read the label file
    with open(label_file, 'r') as f:
      lines = f.readlines()
    # Loop through each line in the label file
    for line in lines:
      # Get the image name from the line
      image_name = line.strip().split()[0]
      # Get the full path of the image
      image_path = [f for f in files if os.path.basename(f) == image_name]
      # Check if the image path exists
      if not image_path:
        raise FileNotFoundError(f"Image {image_name} not found in the input path {input_path}.")
      image_path = image_path[0]
      # Read the image
      image = cv2.imread(image_path)
      # get the left and right coordinates from the line
      coords = line.strip().split()[1:]
      # Convert the coordinates to integers
      coords = [int(float(coord)) for coord in coords]
      left_eye = coords[0:6]
      right_eye = coords[6:12]

      # Extract the label of the image
      label = int(os.path.basename(image_path).split('_')[0])
      # Create a new directory for the label if it does not exist
      label_dir = os.path.join(output_path, str(label))
      if not os.path.exists(label_dir):
        os.makedirs(label_dir)

      # Crop the image to the left and right eye coordinates by extending the center by 32 pixels
      left_eye_image = image[left_eye[3]-32:left_eye[3]+32, left_eye[2]-32:left_eye[2]+32]
      right_eye_image = image[right_eye[3]-32:right_eye[3]+32, right_eye[2]-32:right_eye[2]+32]
      # Save the left eye image
      left_eye_image_path = os.path.join(label_dir, os.path.basename(image_path).replace('.png', '_left.png'))
      right_eye_image_path = os.path.join(label_dir, os.path.basename(image_path).replace('.png', '_right.png'))
      cv2.imwrite(left_eye_image_path, left_eye_image)
      cv2.imwrite(right_eye_image_path, right_eye_image)


if __name__ == "__main__":
  input_path = 'D:\\Workspace\\thesis_sources\\datasets\\gi4e'
  output_path = 'D:\\Workspace\\thesis_sources\\datasets\\gi4egi4e_raw_eyes'
  # Call the main function with the parsed arguments
  main(input_path, output_path)
