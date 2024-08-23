import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('naildisease.h5')

# Define the class labels
class_labels = ['Eczema', 'Leukonychia', 'Onycholysis', 'Pale Nail', 'White Nails']

# Function to predict the class of the image
def predict(image):
    image = load_img(image, target_size=(150, 150))  # Adjust the target size to match the model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

# Streamlit app
st.title("Nail Disease Classification")

st.write("Upload an image of a nail to classify the disease:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Add the Classify button
    if st.button("Classify"):
        st.write("Classifying...")
        label = predict(uploaded_file)
        st.write(f"Predicted Nail Disease: **{label}**")
