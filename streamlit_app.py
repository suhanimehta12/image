
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption


st.set_page_config(page_title="Image Caption Generator", layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F4F8;
        padding: 20px;
    }
    .header {
        font-size: 3em;
        color: #3B3B3B;
        font-weight: bold;
        text-align: center;
    }
    .caption {
        font-size: 1.5em;
        color: #4CAF50;
        font-weight: bold;
        margin-top: 20px;
        text-align: center;
    }
    .button {
        background-color: #FF9800;
        color: white;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.2em;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: 0.3s ease;
        margin-top: 20px;
    }
    .button:hover {
        background-color: #FF5722;
        box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.15);
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .image-container img {
        border-radius: 15px;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    }
    .info-text {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<div class='header'> IMAGE NARRATOR</div>", unsafe_allow_html=True)

st.markdown("""
Used to generate captions for images. Simply upload an image, and the app will provide a detailed description of what is depicted in the image.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='info-text'>Click to generate a description for the image.</div>", unsafe_allow_html=True)


    if st.button("Generate Caption", key="generate_caption", help="Click to generate a caption for the image", use_container_width=True):

        with st.spinner('Generating caption...'):
            time.sleep(2)

            caption = generate_caption(image)

            st.markdown(f"<p class='caption'> Caption: {caption}</p>", unsafe_allow_html=True)

    else:
        st.info("Click the button to generate a caption based on the uploaded image.")
else:
    st.warning("Please upload an image to get started.")

