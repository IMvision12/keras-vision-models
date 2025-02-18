from kvmm.models.inceptionv4 import InceptionV4
import keras

keras.config.set_image_data_format('channels_first')
model = InceptionV4(include_top=True, weights=None, input_shape=(3, 299,299))