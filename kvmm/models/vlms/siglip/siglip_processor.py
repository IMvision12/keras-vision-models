from typing import List, Optional, Union

import keras

from kvmm.models.vlms.siglip.siglip_image_processor import SigLIPImageProcessor
from kvmm.models.vlms.siglip.siglip_tokenizer import SigLIPTokenizer
from kvmm.utils import download_file


@keras.saving.register_keras_serializable(package="kvmm")
class SigLIPProcessor(keras.layers.Layer):
    """
    Combined processor for SigLIP (Sigmoid Loss for Language Image Pre-training) model,
    handling both image and text inputs.

    This processor combines both image processing and text tokenization for SigLIP models.
    It allows you to process both modalities with a single interface, handling all the
    necessary preprocessing steps for SigLIP model inference or training.

    The processor can be customized with various parameters for both the image processor
    and tokenizer components.

    Args:
        image_resolution (int, optional): The target resolution for processed images.
            Default is 224.
        mean (List[float], optional): RGB mean values for image normalization.
            Default is [0.5, 0.5, 0.5].
        std (List[float], optional): RGB standard deviation values for image normalization.
            Default is [0.5, 0.5, 0.5].
        do_center_crop (bool, optional): Whether to apply center cropping to images.
            Default is True.
        do_normalize (bool, optional): Whether to normalize images. Default is True.
        do_resize (bool, optional): Whether to resize images. Default is True.
        vocab_file (str, optional): Path to the vocabulary file for the tokenizer.
            If None, will download the default vocabulary file.
        multilingual (bool, optional): Whether to use multilingual vocabulary.
            Set to True when using multilingual SigLIP models. Default is False.
        context_length (int, optional): Maximum token sequence length. Default is 64.
        do_lower_case (bool, optional): Whether to convert text to lowercase during preprocessing.
            Default is True.
        unk_token (str, optional): Token to use for unknown words. Default is "<unk>".
        pad_token (str, optional): Padding token. Default is "</s>".
        eos_token (str, optional): End of sequence token. Default is "</s>".
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Example:
        ```python
        # Creating a processor with default settings
        processor = SigLIPProcessor()

        # Processing text and images together
        import numpy as np
        from PIL import Image

        # Load an example image
        image = Image.open("example.jpg")
        image_array = keras.utils.img_to_array(image)

        # Process both text and images
        inputs = processor(
            text=["A photo of a cat", "An image of a dog"],
            images=image_array  # Single image or batch of images
        )

        # The result contains both text and image encodings
        print(inputs.keys())  # Contains tokenizer outputs + 'images'

        # Process from file paths
        inputs = processor(
            text=["A photo of a cat"],
            image_paths="path/to/image.jpg"
        )

        # Process multiple images from paths
        inputs = processor(
            text=["Photo 1", "Photo 2"],
            image_paths=["path/to/image1.jpg", "path/to/image2.jpg"]
        )

        # Custom configuration for higher resolution
        high_res_processor = SigLIPProcessor(
            image_resolution=384,
            context_length=128,
            do_lower_case=False
        )

        inputs = high_res_processor(
            text=["A detailed photo of a landscape"],
            images=image_array
        )

        # Using multilingual model
        multilingual_processor = SigLIPProcessor(
            multilingual=True
        )

        inputs = multilingual_processor(
            text=["Une photo d'un chat", "Ein Bild von einem Hund"],
            images=image_array
        )
        ```
    """

    def __init__(
        self,
        # Image processor params
        image_resolution: int = 224,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        # Tokenizer params
        vocab_file: Optional[str] = None,
        multilingual: bool = False,
        context_length: int = 64,
        do_lower_case: bool = True,
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        eos_token: str = "</s>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_processor = SigLIPImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )

        if vocab_file is None:
            if multilingual:
                vocab_file_path = download_file(
                    "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip_multilingual_vocab.model"
                )
            else:
                vocab_file_path = download_file(
                    "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip_vocab.model"
                )
        else:
            vocab_file_path = vocab_file

        self.tokenizer = SigLIPTokenizer(
            vocab_file=vocab_file_path,
            context_length=context_length,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
        )

        self._config = {
            "image_resolution": image_resolution,
            "mean": mean,
            "std": std,
            "do_center_crop": do_center_crop,
            "do_normalize": do_normalize,
            "do_resize": do_resize,
            "vocab_file": vocab_file,
            "context_length": context_length,
            "do_lower_case": do_lower_case,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "eos_token": eos_token,
        }

    def call(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[keras.KerasTensor, List]] = None,
        image_paths: Optional[Union[str, List[str]]] = None,
    ):
        if text is None and images is None and image_paths is None:
            raise ValueError("At least one of 'text', 'images', or 'image_paths' must be provided")
        
        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")
        
        if image_paths is not None:
            if isinstance(image_paths, (list, tuple)) and len(image_paths) == 0:
                raise ValueError("image_paths cannot be an empty list")
            
        encoding = {}

        if text is not None:
            text_encoding = self.tokenizer(inputs=text)
            encoding.update(text_encoding)

        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")

        if images is not None:
            processed_images = self.image_processor(inputs=images)
            encoding["images"] = processed_images

        if image_paths is not None:
            processed_images = self.image_processor(image_paths=image_paths)
            encoding["images"] = processed_images

        if not encoding:
            raise ValueError(
                "Must provide at least one of 'text', 'images', or 'image_paths'"
            )

        return encoding

    def decode_text(
        self, token_ids: keras.KerasTensor, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def get_sequence_length(self, input_ids: keras.KerasTensor) -> keras.KerasTensor:
        return self.tokenizer.get_sequence_length(input_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
