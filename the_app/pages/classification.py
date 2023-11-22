import streamlit as st
import helper.model_helper as mh
import helper.image_helper as ih

# mh.download_file_from_github_release("https://github.com/nnvu-fit/iusai-project/releases/download/v0.1/model_resnes18.pth", "models")

# set page config
st.set_page_config(
    page_title="ResNet18 App",
    page_icon="🧊",
)

st.write("# Welcome to Classification App! 👋")
st.write("This app is used to classify images using ResNet18, ResNet34, DenseNet121 models.")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.subheader("Image")

# Image source
image_source = st.sidebar.radio(
    label="Select image source",
    options=["Upload", "URL"],
    index=0,
    help="Select the image source",
)

# Image
image = None
if image_source == "Upload":
    image = st.sidebar.file_uploader(
        label="Upload image",
        type=["png", "jpg", "jpeg"],
        help="Upload the image to be classified",
    )
else:
    image_url = st.sidebar.text_input(
        label="Enter image URL",
        help="Enter the URL of the image to be classified",
    )
    if image_url != "" or image_url is not None or image_url != None:
        image = ih.download_image(image_url)

# Model source
model_source = st.sidebar.radio(
    label="Select model source",
    options=["Upload", "URL"],
    index=0,
    help="Select the model source",
)

# Classify button
classify_button = st.sidebar.button(
    label="Classify",
    help="Classify the image",
)

# Main
if image is not None:
    st.image(image, caption="Image to be classified", use_column_width=True)
    if classify_button:
        st.write("Classifying...")
        # download model
        model = mh.download_file_from_github_release("https://github.com/nnvu-fit/iusai-project/releases/download/v0.1/model_resnes18.pth")
        # classify image
        result = mh.classify_image(image, model)
else:
    st.write("Please select an image!")

# Footer
st.write("---")

