import keras
from kvmm.models import clip
from kvmm.utils import download_file

class CLIPProcessor(keras.layers.Layer):
    """
    Combined processor for CLIP model, handling both image and text inputs.
    Improved to accept all configuration parameters for both tokenizer and image processor.
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
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_processor = clip.CLIPImageProcessor(
                image_resolution=image_resolution,
                mean=mean,
                std=std,
                do_center_crop=do_center_crop,
                do_normalize=do_normalize,
                do_resize=do_resize
            )

        if vocab_file is None or merges_file is None:
            raise ValueError("When tokenizer is not provided, vocab_file and merges_file must be specified")

        self.tokenizer = clip.CLIPTokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                context_length=context_length,
                errors=errors,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token
            )

    def call(
        self,
        text=None,
        images=None,
        return_tensors=True,
        **kwargs
    ):
        encoding = {}

        if text is not None:
            text_encoding = self.tokenizer(
                texts=text,
                return_tensors=return_tensors
            )
            encoding.update(text_encoding)

        if images is not None:
            image_encoding = self.image_processor(
                images=images,
            )
            encoding.update(image_encoding)

        return encoding