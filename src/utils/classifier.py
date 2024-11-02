import tensorflow as tf
import numpy as np


class Classifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, results):
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
            landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for model input
            prediction = self.model.predict(landmarks)
            return np.argmax(prediction, axis=1)[0]  # Return the class with the highest probability
        return None
