from retinaface import RetinaFace
import os
import cv2
from PIL import Image
def detect_and_save_faces(image_path, output_base_folder):
    # Extract the base name of the original image file and the folder path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    folder_name = os.path.basename(os.path.dirname(image_path))
    
    # Create a directory for each person
    person_folder = os.path.join(output_base_folder, folder_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image '{image_path}'. Please check the file path.")
        return

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces and landmarks
    faces = RetinaFace.detect_faces(rgb_img)

    # Loop through the results and save face images
    for i, key in enumerate(faces.keys()):
        face = faces[key]
        facial_area = face['facial_area']
        x, y, width, height = facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]
        face_img = rgb_img[y:y+height, x:x+width]

        # Convert to PIL image
        face_image = Image.fromarray(face_img)
        face_filename = os.path.join(person_folder, f"{base_name}_face_{i+1}.jpg")
        face_image.save(face_filename)
        print(f"Saved: {face_filename}")

def process_directory(input_base_folder, output_base_folder):
    for root, _, files in os.walk(input_base_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                detect_and_save_faces(image_path, output_base_folder)

if __name__ == "__main__":

    input_base_folder = "/images-small"

    output_base_folder = "images-small-faces"
    
    process_directory(input_base_folder, output_base_folder)