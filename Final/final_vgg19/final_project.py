import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from VGG19 import VGG19
#from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from PIL import Image

content_img_path = 'elephant.jpg'
style_img_path = 'starry_night.jpg'
mixed_img_path = 'white_noise.jpg'

height = 200
width = 200
num_channels = 3

'''
the content representation on layer ‘conv4_2’ 
the style representation on layers ‘conv1_1’, ‘conv2_1’, ‘conv3_1’, ‘conv4_1’ and ‘conv5_1’
'''

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(height, width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpg')

def plot_images(content_img, style_img, mixed_img):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_img / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_img / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_img / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
# plt.show()

def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

def content_loss(session, model, content_img, layers_id):
    layers = []
    for layer_id in layers_id:
        layer = model[layer_id]
        layers.append(layer)
    values = sess.run(layers, feed_dict = {img: content_img})
    # layers = model.get_layer_tensors(layers_id)
    # values = session.run(layers, feed_dict = feed_dict)

    with model.graph.as_default():
        layer_losses = []
        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(layer, value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)
    return total_loss

def gram_matrix(tensor):
    shape = tensor.get_shape()
    channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

def style_loss(session, model, style_img, layers_id):
    #layers = model.get_layer_tensors(layers_id)
    layers = []
    for layer_id in layers_id:
        layer = model[layer_id]
        layers.append(layer)

    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
        values = session.run(gram_layers, feed_dict = {style_img})
        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(layer, value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)
    return total_loss

def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
            tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss

# def style_transfer(content_img, style_img, content_layer_id, 
# 					style_layer_id, alpha = 1.5, beta = 10, num_iter = 100, step_size = 10):
	

content_img = preprocess(content_img_path)
style_img = preprocess(style_img_path)
mixed_img = preprocess(mixed_img_path)

tf.reset_default_graph()

img = tf.placeholder(tf.float32, (1,height,width,num_channels), name='my_original_image')
vgg19 = VGG19(image_shape=(1,height,width,num_channels), input_tensor=img)
vgg19.summary()

output = tf.identity(vgg19['block5_pool'], name='my_output')

# show_graph(tf.get_default_graph().as_graph_def())

with tf.Session() as sess:
    vgg19.load_weights()
    fd = { img: style_img }
    output_val = sess.run(output, fd)

print(output_val.shape, output_val.mean())


# vgg19 = VGG19(weights='imagenet', include_top=False, pooling='avg')
# model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_pool').output)
# keras_output = model.predict(content_img)  # applying VGG preprocessing
# vgg19.summary()
# print(keras_output.shape, keras_output.mean())
