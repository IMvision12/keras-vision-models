from typing import List, Optional, Union

import keras

from kvmm.models.vlms.siglip2.siglip2_image_processor import SigLIP2ImageProcessor
from kvmm.models.vlms.siglip2.siglip2_tokenizer import SigLIP2Tokenizer
from kvmm.utils import download_file


@keras.saving.register_keras_serializable(package="kvmm")
class SigLIP2Processor(keras.layers.Layer):
    """
    Unified processor for SigLIP2 models.

    This processor combines image preprocessing and text tokenization into a single interface
    for SigLIP2 multimodal models. It handles all necessary preprocessing steps for both
    visual and textual inputs, making it easy to prepare data for model inference or training.

    The processor uses a SentencePiece-based tokenizer for text processing and supports
    flexible input formats including PIL Images, numpy arrays, file paths, and various text
    formats. It applies consistent preprocessing including image resizing, normalization,
    and text tokenization with proper padding and truncation.

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
        vocab_file (str, optional): Path to the SentencePiece model file (.spm).
            If None, will download the default SentencePiece model file.
        context_length (int, optional): Maximum token sequence length. Default is 64.
        add_bos (bool, optional): Whether to add beginning of sentence token. Default is False.
        add_eos (bool, optional): Whether to add end of sentence token. Default is False.
        pad_token (str, optional): Padding token. Default is "<pad>".
        bos_token (str, optional): Beginning of sentence token. Default is "<bos>".
        eos_token (str, optional): End of sequence token. Default is "<eos>".
        unk_token (str, optional): Token to use for unknown words. Default is "<unk>".
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Example:
        ```python
        # Creating a processor with default settings
        processor = SigLIP2Processor()

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
        print(inputs.keys())  # Contains 'input_ids' + 'images'

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

        # Custom configuration for higher resolution and longer sequences
        high_res_processor = SigLIP2Processor(
            image_resolution=384,
            context_length=128,
            add_bos=True,
            add_eos=True
        )

        inputs = high_res_processor(
            text=["A detailed photo of a landscape"],
            images=image_array
        )

        # Decode processed text back to strings
        decoded_texts = processor.decode_text(inputs["input_ids"])
        print(decoded_texts)

        # Get actual sequence lengths (excluding padding)
        seq_lengths = processor.get_sequence_length(inputs["input_ids"])
        print(f"Sequence lengths: {seq_lengths}")
        ```

    Note:
        This processor requires the `sentencepiece` library to be installed for text tokenization:
        pip install sentencepiece
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
        context_length: int = 64,
        add_bos: bool = False,
        add_eos: bool = False,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_processor = SigLIP2ImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )

        if vocab_file is None:
            vocab_file_path = download_file(
                "https://github.com/IMvision12/keras-vision-models/releases/download/SigLIP/siglip2_vocab.model"
            )
        else:
            vocab_file_path = vocab_file

        self.tokenizer = SigLIP2Tokenizer(
            vocab_file=vocab_file_path,
            context_length=context_length,
            add_bos=add_bos,
            add_eos=add_eos,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
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
            "add_bos": add_bos,
            "add_eos": add_eos,
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
        }

    def call(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[keras.KerasTensor, List]] = None,
        image_paths: Optional[Union[str, List[str]]] = None,
    ):
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

    def truncate_sequences(
        self, input_ids: keras.KerasTensor, max_length: int
    ) -> keras.KerasTensor:
        return self.tokenizer.truncate_sequences(input_ids, max_length)

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
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

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
