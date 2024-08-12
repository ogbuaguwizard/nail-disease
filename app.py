import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('path/to/your/model.h5')

# Define the class labels (make sure it matches the order used during training)
class_labels = [
    "Darier's disease", "Muehrcke-e's lines", 'Alopecia areata', "Beau's lines",
    'Bluish nail', 'Clubbing', 'Eczema', "Half and half nails (Lindsay's nails)",
    'Koilonychia', 'Leukonychia', 'Onycholysis', 'Pale nail', 'Red lunula',
    'Splinter hemorrhage', "Terry's nail", 'White nail', 'Yellow nails'
]

def preprocess_image(image):
    # Resize and preprocess the image to match the input shape required by your model
    image = image.resize((150, 150))  # Assuming your model input size is 150x150
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction using the loaded model
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    probability = np.max(prediction)
    
    return predicted_class_label, probability

def main():
    st.title("Nail Disease Prediction")
    st.write("Upload an image of a nail to predict the disease.")

    # File uploader to upload images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict the class of the image
        predicted_label, probability = predict_image(image)
        
        # Display the prediction result
        st.write(f"**Predicted Disease:** {predicted_label}")
        st.write(f"**Probability:** {probability:.2f}")

if __name__ == "__main__":
    main()
