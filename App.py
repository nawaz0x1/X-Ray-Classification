# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Streamlit
import streamlit as st

# Class Names
class_names = ['Covid-19', 'Normal', 'Pneumonia']

# Loading the Model in the Cache
@st.cache_resource
def load_cached_model():
    return load_model('Model/model.keras')

model = load_cached_model()

# Prediction Function
def sample_predict(model, image):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction
    predictions = model.predict(img_array)
    confidence_scores = predictions[0]
    predicted_class_index = np.argmax(confidence_scores)
    predicted_class = class_names[predicted_class_index]
    confidence = float(confidence_scores[predicted_class_index])

    return predicted_class, confidence, confidence_scores

# Changing the Background Image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url('https://images.pexels.com/photos/3936358/pexels-photo-3936358.jpeg');
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title('Covid-19 X-Ray Classification')

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

emoji_map = {
    'Covid-19': '😷',
    'Normal': '👌',
    'Pneumonia' : '🤧'
}

# Once Uploaded below block will be executed
if uploaded_file is not None:
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Converting Image to JPG
    image = image.convert('RGB')
    st.image(image, caption='Uploaded Image', width=256)

    # Make prediction
    if st.button('Predict'):
        predicted_class, confidence, confidence_scores = sample_predict(model, image)
        
        color = '#96d491' if predicted_class == 'Normal' else '#ee2400'
        
        # Printing Predicited Class and Confidence Score
        st.markdown(
            f"#### **Predicted Class:** <span style='color:{color};'>{predicted_class} {emoji_map[predicted_class]}</span>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence: {confidence:.2f}")

        # Create a bar chart of confidence scores
        fig, ax = plt.subplots()
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, confidence_scores, color=sns.color_palette("husl", len(class_names)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name}" for name in class_names])
        ax.invert_yaxis()
        ax.set_xlabel('Confidence')
        ax.set_title('Prediction Confidence by Class')
        st.pyplot(fig)