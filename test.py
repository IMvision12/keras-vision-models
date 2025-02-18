from kvmm.models.mix_transformer import MiT_B0
import keras

keras.config.set_image_data_format('channels_first')
model = MiT_B0(include_top=True, weights=None, input_shape=(3, 299,299))