"""
Professional Eye Disease Prediction Application
Predicts eye diseases from uploaded images using a trained CNN model.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import streamlit as st
import cv2
import os
import tempfile
import requests
from io import BytesIO
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['cataract', 'Conjunctivitis', 'swelling', 'Normal', 'Uveitis']
MODEL_URL = "https://raw.githubusercontent.com/1340Rohith/EYE_disease/main/final" # Corrected URL


# Disease information for professional display
DISEASE_INFO = {
    'cataract': 'Clouding of the natural lens of the eye, causing vision impairment',
    'Conjunctivitis': 'Inflammation of the conjunctiva (pink eye), often infectious',
    'swelling': 'Eyelid or periorbital swelling, may indicate inflammation or infection',
    'Normal': 'No apparent pathological condition detected',
    'Uveitis': 'Inflammation of the uvea, the middle layer of the eye'
}

# Exact replica of your original mod1 class
class mod1(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_size,
                      kernel_size=(3,3),
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=(3,3),
                      stride=1,
                      padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=(3,3),
                      stride=1,
                      padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=(3,3),
                      stride=1,
                      padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.conn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size*14*14,
                      out_features=hidden_size*14*14),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size*14*14,
                      out_features=hidden_size*14*14),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size*14*14,
                      out_features=output_size),
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conn(x)
        return x

def create_model(input_size: int = 3, hidden_size: int = 15, output_size: int = 5):
    """Create the exact model architecture that was used for training."""
    return mod1(input_size, hidden_size, output_size)


@st.cache_resource
def load_model():
    """Load model weights from GitHub URL with caching"""
    try:
        logger.info(f"Downloading model from {MODEL_URL}")
        response = requests.get(MODEL_URL)
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
        
        # Load model weights directly from bytes
        model = create_model()
        # Ensure weights_only=False if your PyTorch version is 2.6+ and the file
        # contains more than just state_dict, as previously discussed.
        model.load_state_dict(torch.load(BytesIO(response.content), map_location=DEVICE, weights_only=False)) 
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Convert to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        
        return transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        st.error(f"Image processing error: {str(e)}")
        return None

def predict_with_model(model, image) -> Tuple[str, Dict[str, float]]:
    """Make prediction on the image using the loaded model."""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        if image_tensor is None:
            return None, None
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            probabilities_percent = probabilities * 100
            _, predicted = torch.max(probabilities, 1)
        
        # Get predicted class
        predicted_class = CLASS_NAMES[predicted.item()]
        
        # Create probability dictionary
        prob_dict = {}
        probabilities_array = probabilities_percent.cpu().numpy().flatten()
        
        for i, class_name in enumerate(CLASS_NAMES):
            prob_dict[class_name] = round(float(probabilities_array[i]), 2)
        
        return predicted_class, prob_dict
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def create_streamlit_app():
    """Create the Streamlit web application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Ocular Pathology Analysis System",
        page_icon="⚫",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sophisticated dark theme styling
    st.markdown("""
    <style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        color: #e0e0e0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 300;
        color: #ffffff;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(255,255,255,0.1);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
        border-right: 1px solid #333;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(145deg, #1e1e1e, #0a0a0a);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Prediction results box */
    .prediction-box {
        background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #444;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #666, #999, #666);
    }
    
    /* Probability bars */
    .prob-container {
        margin: 0.8rem 0;
        padding: 0.8rem;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .prob-bar {
        background: linear-gradient(90deg, #333, #555);
        border-radius: 6px;
        height: 12px;
        margin-top: 8px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(90deg, var(--fill-color), var(--fill-color-light));
        box-shadow: 0 0 10px rgba(255,255,255,0.1);
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 400;
        margin: 2rem 0 1rem 0;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* Upload area styling */
    .uploadedFile {
        border: 2px dashed #444;
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #3a3a3a, #2a2a2a);
        border-color: #666;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border: 1px solid #333;
        border-radius: 8px;
        color: #ffffff;
    }
    
    .streamlit-expanderContent {
        background: rgba(0,0,0,0.2);
        border: 1px solid #333;
        border-top: none;
    }
    
    /* Warning and info boxes */
    .stAlert {
        background: rgba(255,255,255,0.05);
        border: 1px solid #444;
        border-radius: 8px;
    }
    
    /* Text styling */
    p, li, span {
        color: #d0d0d0;
        line-height: 1.6;
    }
    
    strong {
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Medical emphasis */
    .medical-emphasis {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        letter-spacing: 0.5px;
    }
    
    /* Confidence indicator */
    .confidence-high { --fill-color: #666; --fill-color-light: #888; }
    .confidence-medium { --fill-color: #555; --fill-color-light: #777; }
    .confidence-low { --fill-color: #444; --fill-color-light: #666; }
    
    /* Elegant instruction styling */
    .instruction-header {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1rem;
        letter-spacing: 0.8px;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
    }
    
    .instruction-item {
        display: flex;
        align-items: flex-start;
        margin: 1rem 0;
        padding: 0.8rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        border-left: 2px solid #666;
        transition: all 0.3s ease;
    }
    
    .instruction-item:hover {
        background: rgba(255,255,255,0.05);
        border-left-color: #888;
        transform: translateX(2px);
    }
    
    .instruction-icon {
        color: #ffffff;
        font-size: 1.2rem;
        margin-right: 0.8rem;
        margin-top: 0.1rem;
        min-width: 24px;
    }
    
    .instruction-text {
        color: #d0d0d0;
        font-size: 0.9rem;
        line-height: 1.5;
        letter-spacing: 0.3px;
    }
    
    .instruction-emphasis {
        color: #ffffff;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">Clinical Eye Disorder Classifier</h1>', 
                unsafe_allow_html=True)
    
    # Preload model when app starts
    if 'model' not in st.session_state:
        with st.spinner("Initializing diagnostic system..."):
            st.session_state.model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="section-header">SYSTEM PARAMETERS</p>', unsafe_allow_html=True)
        
        # Image Acquisition Instructions
        with st.expander("📸 DIRECTIONS", expanded=True):
            st.markdown("""
            
            <div class="instruction-item">
                <div class="instruction-icon">!</div>
                <div class="instruction-text">
                    Position camera at <span class="instruction-emphasis">one arm's distance</span> from the subject for optimal focus and detail capture
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">!</div>
                <div class="instruction-text">
                    Ensure <span class="instruction-emphasis">even, diffused lighting</span> to avoid harsh reflections on the corneal surface
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">!</div>
                <div class="instruction-text">
                    <span class="instruction-emphasis">Crop tightly around the eye</span> - include only the ocular region, excluding surrounding facial features
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">!</div>
                <div class="instruction-text">
                    Maintain <span class="instruction-emphasis">sharp focus</span> on the iris and pupil for accurate pathological assessment
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">!</div>
                <div class="instruction-text">
                    Keep the eye <span class="instruction-emphasis">centrally aligned</span> in the frame with minimal head tilt or rotation
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🔬 DIAGNOSTIC CAPABILITIES", expanded=False):
            for disease, description in DISEASE_INFO.items():
                st.markdown(f"**{disease.upper()}**")
                st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
                st.markdown("---")
        
        with st.expander("⚕️ CLINICAL DISCLAIMERS"):
            st.markdown("""
            <div style="font-size: 0.9rem; line-height: 1.5;">
            • Research prototype - not FDA approved<br>
            • Requires professional medical validation<br>
            • For educational purposes only<br>
            • Clinical correlation recommended
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🧠 NETWORK ARCHITECTURE"):
            st.markdown("""
            <div style="font-size: 0.85rem; font-family: monospace;">
            <strong>DEEP CNN SPECIFICATION:</strong><br>
            ├── Conv2D (3→15) + ReLU + Pool<br>
            ├── Conv2D (15→15) + LeakyReLU + Pool<br>
            ├── Conv2D (15→15) + LeakyReLU + Pool<br>
            ├── Conv2D (15→15) + LeakyReLU + Pool<br>
            └── Dense (2940→2940→2940→5)<br><br>
            <strong>INPUT:</strong> 224×224 RGB<br>
            <strong>OUTPUT:</strong> 5-class probability
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"<div style='text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;'><strong>COMPUTE:</strong> {DEVICE.upper()}</div>", unsafe_allow_html=True)
        
        if st.button("SYSTEM RESET", type="primary"):
            st.session_state.clear()
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.markdown('<p class="section-header">IMAGE ACQUISITION</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select ocular image for analysis",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload high-resolution retinal or anterior segment image"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                st.markdown('<div style="border: 1px solid #444; border-radius: 12px; padding: 1rem; background: rgba(255,255,255,0.02);">', unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, caption="ACQUIRED IMAGE", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Convert to array for processing
                image_array = np.array(image)
                
                # Make prediction
                with st.spinner("PROCESSING NEURAL NETWORK INFERENCE..."):
                    predicted_class, probabilities = predict_with_model(st.session_state.model, image_array)
                
                if predicted_class is None or probabilities is None:
                    st.error("Failed to process image. Please try another image.")
                    return
                
                # Display results in the second column
                with col2:
                    st.markdown('<p class="section-header">DIAGNOSTIC ANALYSIS</p>', unsafe_allow_html=True)
                    
                    # Main prediction
                    confidence = probabilities[predicted_class]
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="medical-emphasis">PRIMARY DIAGNOSIS</div>
                        <h2 style="color: #ffffff; text-align: center; margin: 1rem 0; font-weight: 300; letter-spacing: 1px;">
                            {predicted_class.upper()}
                        </h2>
                        <div style="text-align: center; font-size: 1.2rem; color: #cccccc;">
                            Confidence: <strong style="color: #ffffff;">{confidence:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.markdown('<p class="section-header">PROBABILITY DISTRIBUTION</p>', unsafe_allow_html=True)
                    
                    # Sort probabilities
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (condition, prob) in enumerate(sorted_probs):
                        # Color coding
                        if prob > 50:
                            confidence_class = "confidence-high"
                        elif prob > 20:
                            confidence_class = "confidence-medium"
                        else:
                            confidence_class = "confidence-low"
                        
                        # Rank indicator
                        rank_indicator = "█" if i == 0 else "▊" if i == 1 else "▌"
                        
                        st.markdown(f"""
                        <div class="prob-container">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="color: #ffffff; font-weight: 400; letter-spacing: 0.5px;">
                                    {rank_indicator} {condition.upper()}
                                </span>
                                <span style="color: #ffffff; font-weight: 500; font-size: 1.1rem;">
                                    {prob:.1f}%
                                </span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill {confidence_class}" style="width: {prob}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Clinical information
                    if predicted_class in DISEASE_INFO:
                        st.markdown(f"""
                        <div class="info-box" style="margin-top: 2rem;">
                            <div style="color: #ffffff; font-weight: 400; font-size: 1.1rem; margin-bottom: 0.5rem; letter-spacing: 0.5px;">
                                CLINICAL NOTES
                            </div>
                            <div style="color: #cccccc; line-height: 1.6;">
                                {DISEASE_INFO[predicted_class]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #2d1b1b, #1d0f0f); padding: 1.5rem; border-radius: 12px; border: 1px solid #553333; color: #ffcccc;">
                    <strong>SYSTEM ERROR:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                logger.error(f"Streamlit app error: {e}")

def main():
    """Main application entry point."""
    try:
        create_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main app error: {e}")

if __name__ == "__main__":
    main()
