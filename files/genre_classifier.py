import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class GenreClassifier:
    def __init__(self, model_path, class_names):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)[0]
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))  # 🛠️ <=== FIXED
        return predicted_class, confidence

