import streamlit as st
import helper.model_helper as mh
import helper.image_helper as ih

# the set of models that already saved in this session
models = {}

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

# select model type with selectbox
model_type = st.sidebar.selectbox(
    label="Select model",
    options=mh.model_types,
    index=0,
    help="Select the model",
)

# Classify button
classify_button = st.sidebar.button(
    label="Classify",
    help="Classify the image",
)

# Main
if image is not None:
    st.image(image, caption="Image to be classified", use_column_width=True)

    image_data = None
    if image_source == "Upload":
        image_data = image.read()
    else:
        image_data = image

    if classify_button and (model_type in mh.model_types):
        # get model from model type
        model = None
        if model_type in models:
            model = models[model_type]
        else:
            model = mh.get_model(model_type)
            models[model_type] = model

        if model is None:
            st.write(f"Model {model_type} is not valid!")
        else:
            # classify image
            result = mh.classify_image(image_data, model)
            # show result and score
            st.write(f"Result: {result}")
    st.write("Please select an image!")

# Footer
st.write("---")
