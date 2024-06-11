import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
from PIL import Image

st.title("Image classification")

classNames = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']
img_height = 180
img_width = 180
new_model = tf.keras.models.load_model('fullmodel')


# Function to preprocess the image using TensorFlow utilities
def preprocess_image(image):
    img = image.resize((img_height, img_width))
    # img = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension
    return img_array

# Function to make predictions


def predict_class(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    return score


# Upload images
uploaded_files = st.file_uploader(
    "Choose an image file", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Display predictions for uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(
            image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        # Predict the class of the image
        score = predict_class(image, new_model)

        # Display the prediction result
        st.write(
            f"Predicted Class for {uploaded_file.name}: {classNames[np.argmax(score)]}")
