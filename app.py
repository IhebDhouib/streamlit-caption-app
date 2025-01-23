import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download NLTK resources
nltk.download('punkt')

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess_image(image):
    """Preprocess the image for the model."""
    return transform(image).unsqueeze(0)

# Load model and tokenizer
@st.cache_resource
def load_model():
    """Load the pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

def generate_caption(tokenizer, model):
    """Generate a caption for the uploaded image."""
    input_text = "Example caption generation"  # Placeholder input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    summary_ids = model.generate(input_ids, max_length=50, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit App
st.title("Image Caption Generator")

# Load model
st.write("Loading model...")
tokenizer, model = load_model()
st.success("Model loaded!")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    st.write("Processing the image...")
    preprocessed_image = preprocess_image(image)

    # Generate caption
    st.write("Generating caption...")
    caption = generate_caption(tokenizer, model)

    # Display caption
    st.subheader("Generated Caption:")
    st.write(caption)
