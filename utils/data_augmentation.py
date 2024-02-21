import numpy as np
from random import shuffle
from preprocessor import preprocess_input, _imread as imread, _imresize as imresize, to_categorical
import scipy.ndimage as ndi
import cv2

class ImageGenerator(object):
    def __init__(self, ground_truth_data, batch_size, image_size,
                train_keys, validation_keys,
                ground_truth_transformer=None,
                path_prefix=None,
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
                do_random_crop=False,
                grayscale=False,
                zoom_range=[0.75, 1.25],
                translation_factor=0.3):

        self.ground_truth_data = ground_truth_data
        self.ground_truth_transformer = ground_truth_transformer
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.image_size = image_size
        self.grayscale = grayscale
        self.color_jitter = []

        self._set_var('saturation_var', saturation_var, self.saturation)
        self._set_var('brightness_var', brightness_var, self.brightness)
        self._set_var('contrast_var', contrast_var, self.contrast)

        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_random_crop = do_random_crop
        self.zoom_range = zoom_range
        self.translation_factor = translation_factor

    def _set_var(self, var_name, var, func):
        if var:
            setattr(self, var_name, var)
            self.color_jitter.append(func)

    def _do_random_transform(self, image_array, box_corners=None):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array, box_corners = jitter(image_array, box_corners)

        if self.lighting_std:
            image_array = self.lighting(image_array)

        return image_array, box_corners

    def horizontal_flip(self, image_array, box_corners=None):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            if box_corners is not None:
                box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
        return image_array, box_corners

    def vertical_flip(self, image_array, box_corners=None):
        if np.random.random() < self.vertical_flip_probability:
            image_array = image_array[::-1]
            if box_corners is not None:
                box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
        return image_array, box_corners

    def transform(self, image_array, box_corners=None):
        return self._do_random_transform(image_array, box_corners)

    def preprocess_images(self, image_array):
        return preprocess_input(image_array)

    def flow(self, mode='train'):
        while True:
            keys = self.train_keys if mode == 'train' else self.validation_keys
            shuffle(keys)

            inputs = []
            targets = []

            for key in keys:
                image_path = self.path_prefix + key
                image_array = imread(image_path)
                image_array = imresize(image_array, self.image_size)

                if len(image_array.shape) != 3:
                    continue

                ground_truth = self.ground_truth_data[key]

                if self.do_random_crop:
                    image_array = self._do_random_crop(image_array)

                image_array = image_array.astype('float32')

                if mode in ['train', 'demo']:
                    if self.ground_truth_transformer:
                        image_array, ground_truth = self.transform(image_array, ground_truth)
                        ground_truth = self.ground_truth_transformer.assign_boxes(ground_truth)
                    else:
                        image_array = self.transform(image_array)[0]

                if self.grayscale:
                    image_array = cv2.cvtColor(image_array.astype('uint8'), cv2.COLOR_RGB2GRAY).astype('float32')
                    image_array = np.expand_dims(image_array, -1)

                inputs.append(image_array)
                targets.append(ground_truth)

                if len(targets) == self.batch_size:
                    inputs = np.asarray(inputs)
                    targets = np.asarray(targets)
                    targets = to_categorical(targets)

                    if mode in ['train', 'val']:
                        inputs = self.preprocess_images(inputs)

                    yield self._wrap_in_dictionary(inputs, targets)
                    inputs = []
                    targets = []

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1': image_array}, {'predictions': targets}]
