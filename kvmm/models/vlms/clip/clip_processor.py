import keras

from kvmm.models.vlms.clip.clip_image_processor import CLIPImageProcessor
from kvmm.models.vlms.clip.clip_tokenizer import CLIPTokenizer
from kvmm.utils import download_file


@keras.saving.register_keras_serializable(package="kvmm")
class CLIPProcessor(keras.layers.Layer):
    """
    Combined processor for CLIP model, handling both image and text inputs.

    This processor combines both image processing and text tokenization for CLIP models.
    It allows you to process both modalities with a single interface, handling all the
    necessary preprocessing steps for CLIP model inference or training.

    The processor can be customized with various parameters for both the image processor
    and tokenizer components.

    Args:
        image_resolution (int, optional): The target resolution for processed images.
            Default is 224.
        mean (list, optional): RGB mean values for image normalization.
            Default is [0.48145466, 0.4578275, 0.40821073].
        std (list, optional): RGB standard deviation values for image normalization.
            Default is [0.26862954, 0.26130258, 0.27577711].
        do_center_crop (bool, optional): Whether to apply center cropping to images.
            Default is True.
        do_normalize (bool, optional): Whether to normalize images. Default is True.
        do_resize (bool, optional): Whether to resize images. Default is True.
        vocab_file (str, optional): Path to the vocabulary file for the tokenizer.
            If None, will download the default vocabulary file.
        merges_file (str, optional): Path to the merges file for the tokenizer.
            If None, will download the default merges file.
        context_length (int, optional): Maximum token sequence length. Default is 77.
        errors (str, optional): Error handling strategy for the tokenizer. Default is "replace".
        unk_token (str, optional): Token to use for unknown words. Default is "<|endoftext|>".
        bos_token (str, optional): Beginning of sequence token. Default is "<|startoftext|>".
        eos_token (str, optional): End of sequence token. Default is "<|endoftext|>".
        pad_token (str, optional): Padding token. Default is "<|endoftext|>".
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Example:
        ```python
        # Creating a processor with default settings
        processor = CLIPProcessor()

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
        ```
    """

    def __init__(
        self,
        # Image processor params
        image_resolution=224,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        # Tokenizer params
        vocab_file=None,
        merges_file=None,
        context_length=77,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_processor = CLIPImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )

        if vocab_file is None and merges_file is None:
            vocab_file_path = download_file(
                "https://github.com/IMvision12/keras-vision-models/releases/download/clip/vocab.json"
            )
            merges_file_path = download_file(
                "https://github.com/IMvision12/keras-vision-models/releases/download/clip/merges.txt"
            )
        else:
            vocab_file_path = vocab_file
            merges_file_path = merges_file

        self.tokenizer = CLIPTokenizer(
            vocab_file=vocab_file_path,
            merges_file=merges_file_path,
            context_length=context_length,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

    def call(
        self,
        text=None,
        images=None,
        image_paths=None,
    ):
        encoding = {}

        if text is not None:
            text_encoding = self.tokenizer(inputs=text)
            encoding.update(text_encoding)

        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")

        if images is not None:
            image_encoding = self.image_processor(inputs=images)
            encoding.update(image_encoding)

        if image_paths is not None:
            image_encoding = self.image_processor(image_paths=image_paths)
            encoding.update(image_encoding)

        return encoding
