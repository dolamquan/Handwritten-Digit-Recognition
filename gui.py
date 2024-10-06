import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading it on every interaction
def load_model():
    try:
        model = tf.keras.models.load_model('handwritten_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to grayscale
    image = np.array(image.convert('L'))  # Convert to grayscale
    
    # Resize to 28x28 pixels (as expected by the model)
    image = cv2.resize(image, (28, 28))
    
    # Invert the image colors (so black background, white digit)
    image = np.invert(image)
    
    # Normalize and reshape the image for model input
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Function to predict the digit from the image
def predict_digit(image):
    try:
        prediction = model.predict(image)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Streamlit App Interface
st.title("MNIST Digit Recognition")

# Upload image button
uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    try:
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image for prediction
        processed_image = preprocess_image(image)
        
        # Predict the digit
        prediction = predict_digit(processed_image)

        if prediction is not None:
            # Display the result
            st.write(f"The model predicts this digit is: **{prediction}**")

            # Optionally display the preprocessed image
            fig, ax = plt.subplots()
            ax.imshow(processed_image[0].reshape(28, 28), cmap=plt.cm.binary)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.write("Please upload an image to get a prediction.")
