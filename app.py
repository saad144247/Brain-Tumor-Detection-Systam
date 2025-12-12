import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download # <-- Naya Import
import os

# --- 1. Configuration (Constants) ---
# Yahan aapka confirmed Repo ID use ho raha hai
HF_REPO_ID = "saad1BM/brain-tumor-detection" 
MODEL_FILENAME = "brain_tumor_model.keras" 
MODEL_PATH = MODEL_FILENAME 

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


# --- 2. Model Loading (Fixed using hf_hub_download) ---
@st.cache_resource
def load_trained_model():
    """Trained model ko Hugging Face se download aur load karta hai."""
    try:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000000: # Re-download agar file choti/corrupt ho
            st.info(f"Downloading model from Hugging Face: {HF_REPOID}/{MODEL_FILENAME}")
            
            # hf_hub_download seedha model ko download karega aur uska local path dega
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename=MODEL_FILENAME, 
                cache_dir=".", # Cache ko current directory mein rakhega
            )
            
            # File ko sahi jagah move karein (agar downloaded path alag ho)
            if downloaded_path != MODEL_PATH:
                os.rename(downloaded_path, MODEL_PATH)

            st.success("Model download successful!")

        # Load the model
        model = load_model(MODEL_PATH)
        return model
        
    except Exception as e:
        # st.error(f"Error loading model: {e}")
        st.error(f"Model Load Failed. Ensure 'huggingface-hub' is in requirements.txt and file is public. Error: {e}")
        return None

model = load_trained_model()

# --- Rest of the app.py code remains the same ---
def predict_image(image_file, model):
    # ... (Prediction logic wahi rahegi) ...
    if model is None:
        # Agar model load fail hua, toh yeh error message wapas jaayega
        return "Model Load Failed", 0.0 
    
    # ...
    # PIL image ko array mein convert karein
    img = Image.open(image_file).convert("RGB")
    # ... (baaki prediction logic) ...
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions) * 100
    predicted_class = CLASS_NAMES[predicted_index]
    
    return predicted_class, confidence_score

# --- Streamlit UI code remains the same ---
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.title("🧠 Brain Tumor Detection System (AI Powered)")
st.write("Upload an MRI image below to classify it as one of the tumor types or no tumor.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload MRI Image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
        st.markdown("---")

        if st.button("Detect Tumor"):
            st.spinner("Analyzing image and detecting tumor...")
            
            predicted_class, confidence_score = predict_image(uploaded_file, model)
            
            if predicted_class == 'no_tumor':
                result_label = f"🟢 **Prediction: No Tumor**"
            elif predicted_class == 'Model Load Failed':
                result_label = f"❌ **Prediction: Model Initialization Error**"
                st.error("Model could not be loaded for prediction.")
            else:
                result_label = f"🔴 **Prediction: Tumor ({predicted_class.replace('_', ' ').title()})**"
            
            st.success("✅ Analysis Complete")
            st.subheader(result_label)
            st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
            
            st.write("---")

with col2:
    st.header("Results and Interpretation")
    st.info("The system uses Transfer Learning (VGG16) to classify the image into four categories: Glioma, Meningioma, Pituitary, or No Tumor.")
    
    if uploaded_file is None:
        st.warning("Please upload an image and click 'Detect Tumor' to see the results.")