import cv2
import numpy as np
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, color_mode='grayscale' if grayscale else 'rgb', target_size=target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    return cv2.CascadeClassifier(model_path)

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, scaleFactor=1.3, minNeighbors=5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (int(x + x_offset), int(y + y_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    hsv_values = np.column_stack((np.linspace(0, 1, num_classes), np.ones(num_classes), np.ones(num_classes)))
    colors = (np.array(hsv_values) * 255).astype(np.uint8)
    return colors
