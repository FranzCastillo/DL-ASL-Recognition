import cv2
import mediapipe as mp


class HandProcessor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return image