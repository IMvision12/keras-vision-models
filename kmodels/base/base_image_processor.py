import keras


class BaseImageProcessor(keras.layers.Layer):
    """Abstract base for kmodels image preprocessors.

    Subclasses implement ``call(images)`` returning the model-ready
    pixel tensor (or a dict that includes one). Concrete subclasses
    define their own constructor kwargs (resolution, normalization
    stats, interpolation mode, patch size, etc.) and ``get_config``
    payload — the base intentionally bakes in no defaults.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, images):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(images)`."
        )
