import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np

def load_model():
    model = tf.keras.models.load_model('model/cnn_model.h5')
    return model

def preprocess_image(image_path: Path):
    image_path = str(image_path)  # Ensure image_path is a string
    print(f"Preprocessing image: {image_path}")  # Debug print

    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # Resize to the expected input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(model, image_path: Path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0]
    predicted_class = 'TB' if prediction >= 0.5 else 'Normal'
    probability = float(prediction) if prediction >= 0.5 else 1 - float(prediction)
    return predicted_class, probability

