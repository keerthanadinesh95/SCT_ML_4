
# Sample Usage:
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model('gesture_model.keras')

def predict_gesture(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))
    return list(train_data.class_indices.keys())[np.argmax(pred)]

# Example:
# print(predict_gesture('sample_gesture.jpg'))
