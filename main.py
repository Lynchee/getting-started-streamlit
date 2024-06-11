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
import cv2

from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


NUM_CLASSES = 15
IMG_SIZE = 224

classNames = {0: 'Banana Legs',
              1: 'Beefsteak',
              2: 'Blue Berries',
              3: 'Cherokee Purple',
              4: 'German Orange Strawberry',
              5: 'Green Zebra',
              6: 'Japanese Black Trifele',
              7: 'Kumato',
              8: 'Oxheart',
              9: 'Roma',
              10: 'San Marzano',
              11: 'Sun Gold',
              12: 'Supersweet 100',
              13: 'Tigerella',
              14: 'Yellow Pear'
              }

# Load the trained model
model = tf.keras.models.load_model('tomato.h5')


# Function to preprocess the image using TensorFlow utilitie
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = tf.keras.utils.img_to_array(img)[:, :, :3]

    # Convert to graycsale
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_gray.astype('uint8'),
                      threshold1=30, threshold2=100)  # Canny Edge Detection

    img_array = np.dstack((img_array, edges[:, :, np.newaxis]))

    img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension
    return img_array, edges


# Function to make predictions
def predict_class(processed_image, model):
    predictions = model.predict(processed_image, verbose=0)
    score = np.argmax(predictions[0])
    return score


# Upload images
uploaded_files = st.file_uploader(
    "Choose an image file", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Display predictions for uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        # st.image(
        #     image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        # Preprocessing image
        processed_image, edges = preprocess_image(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        with col2:
            st.image(
                edges, caption=f"Edge Image", use_column_width=True)

        with col3:
            st.image(
                processed_image.numpy()/255.0, caption=f"CH4 Image", use_column_width=True)

        # Predict the class of the image
        score = predict_class(processed_image, model)

        # Display the prediction result
        st.write(
            f"Predicted Class: {classNames[score]}")
