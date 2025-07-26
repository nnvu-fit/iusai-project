
import os
import glob
import re


def main(input_path, output_path, num_workers=4):
  """
  Main function to process the NuScenes dataset for hand detection pre-training.
  Args:
      input_path (str): Path to the NuScenes dataset.
      output_path (str): Path to save the processed data.
      num_workers (int): Number of workers for data processing.
  """

  hand_images = glob.glob(os.path.join(input_path, 'Hand Postures', '*.jpg'), recursive=True)

  if not hand_images:
    raise FileNotFoundError(f"No hand images found in the input path {input_path}.")
  # Create the output directory if it does not exist
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  max_workers = num_workers
  if max_workers <= 0:
    raise ValueError("Number of workers must be greater than 0.")
  
  filename_regex = r'^\w \(\d+\).jpg$' # Example regex for filenames like "A (123).jpg"
  tasks = []
  for image_path in hand_images:
    # Extract the label from the image filename
    filename = os.path.basename(image_path)

    if not re.match(filename_regex, filename):
      continue # Skip files that do not match the regex
    
    print(f"Processing {filename}...")

    label = filename.split(' ')[0]  # Extract the label from the filename
    label_dir = os.path.join(output_path, label)

    if not os.path.exists(label_dir):
      os.makedirs(label_dir)
    # Copy the image to the label directory
    output_image_path = os.path.join(label_dir, filename)
    if not os.path.exists(output_image_path):
      os.rename(image_path, output_image_path)  # Move the file to the new location
    else:
      print(f"File {output_image_path} already exists. Skipping.")  
  print(f"Processed {len(hand_images)} hand images and saved to {output_path}.")


if __name__ == "__main__":
  input_path = './data/nus2handssign'
  output_path = './datasets/nus2hands'

  main(input_path, output_path)
