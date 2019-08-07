import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import functools

# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # (384,512,3), 0~1

    # scale
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]  # add batch axis
    return img


# load img
content_path = os.path.join("img", 'turtle.jpg')
style_path = os.path.join("img", 'kandinsky.jpg')
content_image = load_img(content_path)
style_image = load_img(style_path)


# Gram matrix calculation to compute style
def gram_matrix(input_tensor):
    input_shape = input_tensor.get_shape()
    input_reshape = tf.reshape(input_tensor, (input_shape[0], input_shape[1] * input_shape[2], input_shape[3]))
    gram = tf.matmul(input_reshape, input_reshape, transpose_a=True)  # (batch, channel, channel)
    return gram / tf.cast(input_shape[1] * input_shape[2], tf.float32)


# build a model to return style and content tensor
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pre-trained VGG, trained on image-net data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(n).output for n in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layer, content_layer):
        super(StyleContentModel, self).__init__()
        # VGG model
        self.vgg = vgg_layers(style_layer + content_layer)
        # parameter
        self.style_layers = style_layer
        self.content_layers = content_layer
        self.num_style_layers = len(style_layer)
        self.vgg_trainable = False

    def call(self, inputs):
        # 预处理图片
        inputs = inputs * 255.
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # VGG中提取特征. outputs中包括了style和content提取的特征
        outputs = self.vgg(preprocessed_input)

        # style_outputs是一个list，含5个元素，每个元素是(batch,w,h,channels)的tensor. content_outputs仅含1个元素
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        # 对 style_outputs 计算 gram 矩阵.
        # 计算前：shape: [(1, 384, 512, 64), (1, 192, 256, 128), (1, 96, 128, 256), (1, 48, 64, 512), (1, 24, 32, 512)]
        # 计算后: shape: [(1, 64, 64), (1, 128, 128), (1, 256, 256), (1, 512, 512), (1, 512, 512)]
        # Gram矩阵的计算是按照通道将特征拉开，随后进行矩阵乘法.
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # 输出
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {"content": content_dict, "style": style_dict}


# layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# 调用函数，提取特征
extractor = StyleContentModel(style_layers, content_layers)
# results包含"style"和"content"两个键, 每个键包含content_name和value, 其中value是矩阵
# results = extractor(tf.constant(content_image))

# target 计算
style_targets = extractor(style_image)['style']  # 提取 style_image 的 style 特征
content_targets = extractor(content_image)['content']  # 提取 content_image 的 content 特征

# 给定一张随机原始图像，迭代该图像
image = tf.Variable(content_image)  # 原始图像应该是一张随机图，为了快速迭代初始化为content image
optimizer = tf.train.AdamOptimizer(learning_rate=0.02, beta1=0.99, epsilon=0.1)

# loss weight.  由于初始化为content_image, 因此content损失很小
style_weight, content_weight, total_variation_weight = 0.01, 1e4, 1e8


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()]) * style_weight / len(style_layers)
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()]) * content_weight / len(content_layers)
    loss_all = style_loss, content_loss
    return style_loss, content_loss, loss_all


def total_variation_loss(img):
    """ 图像总变差作为约束(类似梯度sobel算子). 使生成的图像变化较为平滑
            Decrease these using an explicit regularization term on the high frequency components
            of the image. In style transfer, this is often called the total variation loss:
    """
    x_deltas = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_deltas = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


def train_step(img):
    """ 进行一步迭代，更新图片 """
    with tf.GradientTape() as tape:
        outputs = extractor(img)
        style_loss, content_loss, loss = style_content_loss(outputs)
        tv_loss = total_variation_weight * total_variation_loss(img)
        loss += tv_loss
        print("style loss:", style_loss.numpy(), ", content_loss:", content_loss.numpy(), ", tv_loss:", tv_loss.numpy())
    grad = tape.gradient(loss, img)
    optimizer.apply_gradients([(grad, img)])  # 更新 image
    # 像素范围不超过 0~1
    img.assign(tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0))


# 循环迭代
for n in range(10):
    for m in range(100):
        train_step(image)
    plt.imshow(image.numpy()[0])
    filename = os.path.join("result", "train_" + str(n))
    plt.title(filename)
    plt.savefig(filename + ".jpg")
