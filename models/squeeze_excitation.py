import keras.layers
import keras.backend

def se_block(input, ratio=16):
    """
    Implementation of a squeeze-and-excitation block.

    :param input: The input of the squeeze-and-excitation block.
    :param ratio: The reduction ratio of the squeeze-and-excitation block. According to the paper, the best results are
    achieved for ratio=16
    :return:
    """

    if keras.backend.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = -1

    channels = input.keras_shape[axis]

    x = keras.layers.GlobalAveragePooling2D()(input)
    x = keras.layers.Dense(channels//ratio, activation='relu')(x)
    x = keras.layers.Dense(channels//ratio, activation='sigmoid')(x)
    x = keras.layers.multiply()[input, x]

    return x