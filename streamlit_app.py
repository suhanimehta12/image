import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time

# Initialize device (to handle large models on CPU or GPU)
device = torch.device("cpu")  # Switch to "cuda" if using GPU

def load_model_with_retry():
    retry_count = 0
    while retry_count < 3:  # Try 3 times
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
            return processor, model
        except Exception as e:
            retry_count += 1
            st.error(f"Failed to load model. Attempt {retry_count}/3. Error: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
    st.error("Model loading failed after 3 attempts.")
    return None, None

processor, model = load_model_with_retry()

def generate_caption(image):
    if processor is None or model is None:
        st.error("Model is not loaded properly. Please try again later.")
        return ""
    inputs = processor(image, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

# Streamlit app configuration
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
    st.warning("Please upload an image to get started.")


