import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.datasets import get_labels
from keras.models import load_model
import pickle

def make_mosaic(images, num_rows, num_cols, border=1):
    num_images, *image_shape = images.shape
    mosaic = np.ma.masked_all((num_rows * (image_shape[0] + border) - border,
                               num_cols * (image_shape[1] + border) - border),
                               dtype=np.float32)
    for i, image in enumerate(images):
        row, col = divmod(i, num_cols)
        row_start, col_start = row * (image_shape[0] + border), col * (image_shape[1] + border)
        mosaic[row_start:row_start + image_shape[0], col_start:col_start + image_shape[1]] = np.squeeze(image)
    return mosaic

def pretty_imshow(axis, data, vmin=None, vmax=None, cmap=cm.jet):
    vmin = vmin or data.min()
    vmax = vmax or data.max()
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    image = axis.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(image, cax=cax)

def display_image(face, class_vector=None, class_decoder=None, pretty=False):
    if class_vector is not None and class_decoder is None:
        raise Exception('Provide class decoder')
    face = np.squeeze(face)
    color_map = 'gray' if len(face.shape) < 3 else None
    plt.figure()
    if class_vector is not None:
        class_name = class_decoder[np.argmax(class_vector)]
        plt.title(class_name)
    if pretty:
        pretty_imshow(plt.gca(), face, cmap=color_map)
    else:
        plt.imshow(face, cmap=color_map)

def draw_mosaic(data, num_rows, num_cols, class_vectors=None, class_decoder=None, cmap='gray'):
    if class_vectors is not None and class_decoder is None:
        raise Exception('Provide class decoder')

    fig, axes = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(8, 8, forward=True)
    titles = [class_decoder[np.argmax(class_vector)] for class_vector in class_vectors] if class_vectors else []

    for ax, image, title in zip(axes.flat, data, titles):
        image = np.squeeze(image)
        ax.axis('off')
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
    plt.tight_layout()

if __name__ == '__main__':
    dataset_name = 'fer2013'
    class_decoder = get_labels(dataset_name)
    faces = pickle.load(open('faces.pkl', 'rb'))
    emotions = pickle.load(open('emotions.pkl', 'rb'))

    pretty_imshow(plt.gca(), make_mosaic(faces[:4], 2, 2), cmap='gray')
    plt.show()

    num_images_to_show = 4
    pretty_imshow(plt.gca(), make_mosaic(faces[:num_images_to_show], 2, 2), cmap='gray')
    plt.show()

    model = load_model('../trained_models/emotion_models/simple_CNN.985-0.66.hdf5')
    conv1_weights = np.squeeze(model.layers[2].get_weights()[0])
    conv1_weights = np.rollaxis(conv1_weights, 2, 0)[..., np.newaxis]
    num_kernels = conv1_weights.shape[0]
    box_size = int(np.ceil(np.sqrt(num_kernels)))

    print('Box size:', box_size)
    print('Kernel shape', conv1_weights.shape)

    plt.figure(figsize=(15, 15))
    plt.title('conv1 weights')
    pretty_imshow(plt.gca(), make_mosaic(conv1_weights, box_size, box_size), cmap=cm.binary)
    plt.show()
