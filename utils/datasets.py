from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

class DataManager:
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path or {
            'imdb': '../datasets/imdb_crop/imdb.mat',
            'fer2013': '../datasets/fer2013/fer2013.csv',
            'KDEF': '../datasets/KDEF/'
        }.get(dataset_name, None)
        self.image_size = image_size
    
    def get_data(self):
        loader_method = getattr(self, f'_load_{self.dataset_name}', lambda: None)
        return loader_method()

    def _load_imdb(self):
        face_score_threshold = 3
        dataset = loadmat(self.dataset_path)
        data = dataset['imdb']
        mask = (data['face_score'] > face_score_threshold) & np.isnan(data['second_face_score']) & ~np.isnan(data['gender'][0, 0][0])
        image_names = [name[0] for name in data['full_path'][0, 0][0][mask]]
        gender_classes = data['gender'][0, 0][0][mask].tolist()
        return dict(zip(image_names, gender_classes))

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        faces = [cv2.resize(np.asarray([int(pixel) for pixel in face.split(' ')]).reshape(48, 48).astype('float32'), self.image_size) for face in data['pixels']]
        return dict(faces=np.asarray(faces).reshape(-1, *self.image_size, 1), emotions=pd.get_dummies(data['emotion']).values)

    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)
        file_paths = [os.path.join(folder, filename) for folder, _, filenames in os.walk(self.dataset_path) for filename in filenames if filename.lower().endswith(('.jpg'))]
        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))

        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), (y_size, x_size))
            faces[file_arg] = image_array
            file_emotion = os.path.basename(file_path)[4:6]

            try:
                emotion_arg = class_to_arg[file_emotion]
                emotions[file_arg, emotion_arg] = 1
            except KeyError:
                continue

        return dict(faces=np.expand_dims(faces, -1), emotions=emotions)

def get_labels(dataset_name):
    return {
        'fer2013': {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'},
        'imdb': {0:'woman', 1:'man'},
        'KDEF': {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    }.get(dataset_name, None)

def get_class_to_arg(dataset_name='fer2013'):
    return {
        'fer2013': {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6},
        'imdb': {'woman':0, 'man':1},
        'KDEF': {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    }.get(dataset_name, None)

def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle:
        shuffle(ground_truth_keys)
    num_train = int((1 - validation_split) * len(ground_truth_keys))
    return ground_truth_keys[:num_train], ground_truth_keys[num_train:]

def split_data(x, y, validation_split=.2):
    num_train_samples = int((1 - validation_split) * len(x))
    return (x[:num_train_samples], y[:num_train_samples]), (x[num_train_samples:], y[num_train_samples:])
