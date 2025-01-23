import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
@st.cache_resource
def load_blip_model():
    """Load BLIP model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    """Generate a caption for the given image using BLIP."""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_beams=5)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit App
st.title("Image Caption Generator")

# Load model
st.write("Loading BLIP model...")
processor, model = load_blip_model()
st.success("BLIP model loaded!")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate caption
    st.write("Generating caption...")
    caption = generate_image_caption(image, processor, model)

    # Display caption
    st.subheader("Generated Caption:")
    st.write(caption)
