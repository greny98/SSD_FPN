from tensorflow.keras import layers, Model
from tensorflow.keras.applications import mobilenet_v2

from configs.common_config import IMAGE_SIZE


def get_backbone(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    base_net = mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    extract_layers = ['block_4_add', 'block_8_add', 'block_15_add']
    # base_net = densenet.DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    # extract_layers = ['pool2_pool', 'pool3_pool', 'pool4_pool']
    feature_maps = [base_net.get_layer(name).output for name in extract_layers]
    return Model(inputs=[base_net.inputs], outputs=feature_maps)


def pyramid_block(l_layers):
    out_layers = []
    for i in range(len(l_layers) - 2, -1, -1):
        upscale = layers.UpSampling2D(2)(l_layers[i + 1])
        out = layers.Add()([l_layers[i], upscale])
        out_layers.append(out)
    out_layers = out_layers[::-1]
    out_layers.append(l_layers[-1])
    return out_layers


def FeaturePyramid(backbone: Model, filters=128):
    pool_out1, pool_out2, pool_out3 = backbone.outputs
    # Change all to 256 units
    pyr_out1 = layers.Conv2D(filters, 1, name='pyr_out1_conv1')(pool_out1)
    pyr_out2 = layers.Conv2D(filters, 1, name='pyr_out2_conv1')(pool_out2)
    pyr_out3 = layers.Conv2D(filters, 1, name='pyr_out3_conv1')(pool_out3)
    # pyramid handle
    pyr_out1, pyr_out2, pyr_out3 = pyramid_block(
        [pyr_out1, pyr_out2, pyr_out3])
    # after pyramid
    pyr_out1 = layers.Conv2D(filters, 3, 1, padding='same', name='pyr_out1_conv2')(pyr_out1)
    pyr_out2 = layers.Conv2D(filters, 3, 1, padding='same', name='pyr_out2_conv2')(pyr_out2)
    pyr_out3 = layers.Conv2D(filters, 3, 1, padding='same', name='pyr_out3_conv2')(pyr_out3)
    pyr_out4 = layers.Conv2D(filters, 3, 2, "same", name='pyr_out4_conv2')(pyr_out3)
    pyr_out5 = layers.Conv2D(filters, 3, 2, "same", name='pyr_out5_conv2')(layers.ReLU()(pyr_out4))
    return Model(inputs=[backbone.inputs],
                 outputs=[pyr_out1, pyr_out2, pyr_out3, pyr_out4, pyr_out5])
