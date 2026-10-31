"""Model loading, OpenCV preprocessing, prediction and Grad-CAM for the defect detection app."""
import json
import os
import threading
import time

import cv2
import keras
import numpy as np
import tensorflow as tf


class DefectDetector:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'model_config.json')) as f:
            self.config = json.load(f)
        self.model = keras.models.load_model(os.path.join(model_dir, 'defect_model.keras'))
        self.img_size = tuple(self.config['img_size'])
        self.threshold = float(self.config['threshold'])
        self.class_names = self.config['class_names']          # ['OK', 'DEFECTIVE']

        # Layers used to run the model in pieces, so one pass gives both the prediction and Grad-CAM
        self._rescale = self.model.get_layer('rescale')
        self._base = self.model.get_layer(self.config['base_model_layer'])
        self._gap = self.model.get_layer('gap')
        self._pred = self.model.get_layer('pred')
        self._lock = threading.Lock()

        self.predict(np.zeros(self.img_size + (3,), np.uint8))  # warm-up so the first real request is fast

    def preprocess(self, image_bgr):
        """BGR image (cv2.imread / webcam frame) -> float32 batch (1, H, W, 3), raw RGB 0-255."""
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(img.astype('float32'), axis=0)

    def predict(self, image_bgr):
        x = self.preprocess(image_bgr)
        start = time.perf_counter()
        with self._lock:
            conv_out = self._base(self._rescale(x), training=False)
            with tf.GradientTape() as tape:
                tape.watch(conv_out)
                pred = self._pred(self._gap(conv_out))
                score = pred[:, 0]
            grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        heatmap = (tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000

        defect_prob = float(pred[0, 0])
        is_defective = defect_prob >= self.threshold
        return {
            'label': self.class_names[int(is_defective)],
            'is_defective': bool(is_defective),
            'defect_prob': defect_prob,
            'confidence': defect_prob if is_defective else 1 - defect_prob,
            'inference_ms': elapsed_ms,
            'heatmap': heatmap,
        }


def overlay_heatmap(image_bgr, heatmap, alpha=0.4):
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, hm, alpha, 0)


def decode_image(data):
    """Raw file bytes -> BGR image, or None if the bytes are not a valid image."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
