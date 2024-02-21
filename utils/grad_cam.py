import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import h5py
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda
from keras.models import load_model, Sequential
from preprocessor import preprocess_input

def reset_optimizer_weights(model_filename):
    with h5py.File(model_filename, 'r+') as model:
        del model['optimizer_weights']

def target_category_loss(x, category_index, num_classes):
    return tf.multiply(x, K.one_hot([category_index], num_classes))

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(image_array):
    return preprocess_input(np.expand_dims(image_array, axis=0))

def register_gradient():
    if "GuidedBackProp" not in tf.compat.v1.get_default_graph().get_all_collection_keys():
        @tf.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, gradient):
            dtype = op.inputs[0].dtype
            guided_gradient = (gradient * tf.cast(gradient > 0., dtype) *
                               tf.cast(op.inputs[0] > 0., dtype))
            return guided_gradient

def compile_saliency_function(model, activation_layer='conv2d_7'):
    input_image = model.input
    layer_output = model.get_layer(activation_layer).output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_image)[0]
    return K.function([input_image, K.learning_phase()], [saliency])

def modify_backprop(model, name, task):
    graph = tf.compat.v1.get_default_graph()

    with graph.gradient_override_map({'Relu': name}):
        for layer in model.layers:
            if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
                layer.activation = tf.nn.relu

        model_path = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5' if task == 'gender' else '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        new_model = load_model(model_path, compile=False)

    return new_model

def deprocess_image(x):
    x = (x - x.mean()) / (x.std() + 1e-5)
    x = x * 0.1 + 0.5
    x = np.clip(x, 0, 1) * 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    return np.clip(x, 0, 255).astype('uint8')

def compile_gradient_function(input_model, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    num_classes = model.output_shape[1]

    def target_category_loss(x, category_index, num_classes):
        return tf.multiply(x, K.one_hot([category_index], num_classes, on_value=1.0, off_value=0.0, dtype=tf.float32))

    target_layer = Lambda(lambda x: target_category_loss(x, category_index, num_classes),
                         output_shape=lambda x: (x[0], num_classes))
    model.add(target_layer)

    loss = K.sum(model.layers[-1].output)
    conv_output = model.layers[0].get_layer(layer_name).output
    gradients = normalize(K.gradients(loss, conv_output)[0])

    gradient_function = K.function([model.layers[0].input, K.learning_phase()],
                                   [conv_output, gradients])

    return gradient_function

def calculate_gradient_weighted_CAM(gradient_function, image):
    output, evaluated_gradients = gradient_function([image, False])
    
    output, evaluated_gradients = output[0, :], evaluated_gradients[0, :, :, :]
    weights = np.mean(evaluated_gradients, axis=(0, 1))

    CAM = np.dot(output, weights)
    CAM = np.maximum(CAM, 0)
    heatmap = CAM / np.max(CAM)

    image = image[0, :]
    image = np.clip(image - np.min(image), 0, 255)

    CAM = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    CAM = np.float32(CAM) + np.float32(image)
    CAM = 255 * CAM / np.max(CAM)
    
    return np.uint8(CAM), heatmap

def calculate_guided_gradient_CAM(preprocessed_input, gradient_function, saliency_function):
    CAM, heatmap = calculate_gradient_weighted_CAM(gradient_function, preprocessed_input)
    saliency = saliency_function([preprocessed_input, 0])
    gradCAM = saliency[0] * heatmap[..., np.newaxis]
    return deprocess_image(gradCAM)

def calculate_guided_gradient_CAM_v2(preprocessed_input, gradient_function,
                                    saliency_function, target_size=(128, 128)):
    CAM, heatmap = calculate_gradient_weighted_CAM(gradient_function, preprocessed_input)
    
    heatmap = cv2.resize(heatmap.astype('uint8'), target_size)
    
    saliency = saliency_function([preprocessed_input, 0])[0, ..., 0]
    saliency = cv2.resize(saliency.astype('uint8'), target_size)
    
    gradCAM = saliency * heatmap
    gradCAM = deprocess_image(gradCAM)
    
    return np.expand_dims(gradCAM, -1)

if __name__ == '__main__':
    import pickle
    faces = pickle.load(open('faces.pkl', 'rb'))
    face = faces[0]
    model_filename = '../../trained_models/emotion_models/mini_XCEPTION.523-0.65.hdf5'
    reset_optimizer_weights(model_filename)
    model = load_model(model_filename)

    preprocessed_input = load_image(face)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    gradient_function = compile_gradient_function(model, predicted_class, 'conv2d_6')
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', 'emotion')
    saliency_function = compile_saliency_function(guided_model)
    guided_gradCAM = calculate_guided_gradient_CAM(preprocessed_input,
                                gradient_function, saliency_function)

    cv2.imwrite('guided_gradCAM.jpg', guided_gradCAM)
