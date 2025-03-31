import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to generate caption
def generate_caption(image, max_length=30):
    inputs = processor(image, return_tensors="pt")
    caption_ids = model.generate(**inputs, max_length=max_length)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

# Streamlit UI
st.title("Image Caption Generator")
st.write("Upload an image to generate a descriptive caption.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    if st.button("Generate Caption"):
        caption = generate_caption(image, max_length=20)
        st.success(f"**Generated Caption:** {caption}")

