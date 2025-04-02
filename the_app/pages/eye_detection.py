import streamlit as st
import helper.image_helper as ih
import models.faster_rcnn as FasterRCNN

st.set_page_config(
    page_title="Eye Detection App",
    page_icon="👁️",
)

# toasts
toast = st.toast
# sidebar
sidebar = st.sidebar

st.write("# Welcome to Eye Detection App! 👁️")
st.write("This app is used to detect eye using Faster RCNN.")

st.write("## How to use this app?")
st.write(
    """
    1. From left pannel - Upload an image of a person.
    2. From left pannel - Click on the "Detect Eyes" button.
    3. The app will display the image with detected eyes highlighted.
    """
)
sidebar.write("## Upload Image")
image = sidebar.file_uploader(
    label="Upload image",
    type=["png", "jpg", "jpeg"],
    help="Upload the image to be detected",
)
sidebar.write("## Detect Eyes")
detect_button = sidebar.button(
    label="Detect Eyes",
    help="Detect eyes in the image",
)

# Display the uploaded image
if image is not None:
    st.write("## Image Preview")
    image_container = st.image(image, caption="Uploaded Image", use_container_width=True)


if detect_button:
    if image is not None:
        # Load the image
        image = image.read()
        # Detect eyes in the image
        image, detected_image = FasterRCNN.detect_eyes(image)
        # Display the detected image by replacing the image in the container
        image_container.image(image, caption="Detected Eyes", use_container_width=True)

        # Display the detected image
        st.write("## Detected Eyes")
        st.image(detected_image, caption="Detected Eyes", use_container_width=True)


st.write("## About")
st.write(
    """
    This app is built using Streamlit and uses Faster RCNN for eye detection.
    The model is trained on a dataset of images with annotated eyes.
    """
)
