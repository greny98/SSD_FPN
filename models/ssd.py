from tensorflow.keras import layers, Model, regularizers
from models.feature_pyramid import get_backbone, FeaturePyramid

l2 = regularizers.l2(1.5e-5)


def build_head(feature, name):
    for i in range(4):
        feature = layers.Conv2D(128, 3, padding="same", name=name + '_conv' + str(i))(feature)
        feature = layers.BatchNormalization(epsilon=1.001e-5)(feature)
        feature = layers.ReLU()(feature)
    return feature


def ssd_head(features):
    classes_outs = []
    box_outputs = []
    for idx, feature in enumerate(features):
        classify_head = build_head(feature, 'classify_head' + str(idx))
        detect_head = build_head(feature, 'detect_head' + str(idx))
        box_outputs.append(detect_head)
        classes_outs.append(classify_head)
    return classes_outs, box_outputs


def create_ssd_model(num_classes, base_ckpt=None):
    backbone = get_backbone()
    pyramid = FeaturePyramid(backbone)
    classes_heads, box_heads = ssd_head(pyramid.outputs)

    num_anchor_boxes = 9
    classes_outs = []
    box_outputs = []
    for idx, head in enumerate(box_heads):
        detect_head = layers.Conv2D(num_anchor_boxes * 4, 3, padding="same",
                                    name='detect_head' + str(idx) + '_conv_out',
                                    kernel_regularizer=l2)(head)
        box_outputs.append(layers.Reshape([-1, 4])(detect_head))

    head_model = Model(inputs=[pyramid.input], outputs=[classes_heads + box_heads + box_outputs])
    if base_ckpt is not None:
        head_model.load_weights(base_ckpt).expect_partial()

    for idx, head in enumerate(classes_heads):
        classify_head = layers.Conv2D(num_anchor_boxes * num_classes, 3, padding="same",
                                      name='classify_head' + str(idx) + '_conv_out',
                                      kernel_regularizer=l2)(head)
        classes_outs.append(layers.Reshape([-1, num_classes])(classify_head))

    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    box_outputs = layers.Concatenate(axis=1)(box_outputs)
    outputs = layers.Concatenate(axis=-1)([box_outputs, classes_outs])
    return Model(inputs=[backbone.input], outputs=[outputs])
