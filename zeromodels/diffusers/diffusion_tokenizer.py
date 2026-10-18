import keras
from keras import ops

from zeromodels.base import BaseTokenizer
from zeromodels.models.clip.clip_tokenizer import CLIPTokenizer
from zeromodels.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from zeromodels.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from zeromodels.models.t5.t5_tokenizer import T5Tokenizer

QWEN_IMAGE_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
QWEN_IMAGE_21_PROMPT_TEMPLATE = (
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Per model_type: the tokenizer backend and its defaults. "clip": the CLIP BPE
# tokenizer padded to a fixed length (SDXL and SD 3 derive their second tower's
# "!"-padded ids from its mask; SD 3 adds the T5 tokenizer from tokenizer_3.json).
# "qwen": a Qwen BPE tokenizer wrapping each prompt in the pipeline's ChatML
# template (the template prefix is dropped after the text encoder).
DIFFUSION_TOKENIZERS = {
    "stable_diffusion": {"kind": "clip", "pad_token": "<|endoftext|>"},
    "stable_diffusion_2": {"kind": "clip", "pad_token": "!"},
    "stable_diffusion_xl": {"kind": "clip", "pad_token": "<|endoftext|>"},
    "stable_diffusion_xl_refiner": {"kind": "clip", "pad_token": "<|endoftext|>"},
    "stable_diffusion_3": {"kind": "clip", "pad_token": "<|endoftext|>", "t5": True},
    "stable_diffusion_3_5": {
        "kind": "clip",
        "pad_token": "<|endoftext|>",
        "t5": True,
    },
    "qwen_image": {
        "kind": "qwen",
        "tokenizer_class": Qwen2Tokenizer,
        "prompt_template": QWEN_IMAGE_PROMPT_TEMPLATE,
        "prompt_template_start_idx": 34,
        "empty_prompt": "",
    },
    "qwen_image_21": {
        "kind": "qwen",
        "tokenizer_class": Qwen3Tokenizer,
        "prompt_template": QWEN_IMAGE_21_PROMPT_TEMPLATE,
        "prompt_template_start_idx": 14,
        "empty_prompt": " ",
    },
}
DEFAULT_MAX_SEQ_LEN = {"clip": 77, "qwen": 1024}


@keras.saving.register_keras_serializable(package="zeromodels")
class ZMDiffusionTokenizer(BaseTokenizer):
    """The prompt tokenizer of every :class:`ZMDiffusion` checkpoint.

    One class for all diffusion families, selected by ``model_type`` like the model:
    loading by repo id reads it from the repo (``zm_preprocessor.json`` when present,
    else ``zm_config.json``) together with the repo's ``tokenizer.json``::

        tokenizer = ZMDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
        image = pipe.generate(**tokenizer("a photo of a cat"))

    - Stable Diffusion 1.x / XL (base and refiner): CLIP ViT-L/14's byte-pair
      encoding with ``<|startoftext|>`` / ``<|endoftext|>`` framing, truncated and
      ``<|endoftext|>``-padded to 77 tokens. SDXL's second text tower pads with
      ``!`` instead; :class:`ZMDiffusion` derives those ids from the returned
      ``attention_mask``, so one call feeds ``generate``.
    - Stable Diffusion 2.x: the same encoding padded with ``!`` (id 0), the way
      OpenCLIP ViT-H/14 was trained.
    - Stable Diffusion 3 / 3.5: the CLIP tokens plus the T5 SentencePiece tokens of
      the repo's ``tokenizer_3.json`` (``</s>``-terminated, ``<pad>``-padded to
      ``max_sequence_length``, 256) as ``input_ids_3``.
    - Qwen-Image / Qwen-Image-2.1: each prompt wrapped in the pipeline's ChatML
      template, then Qwen2 / Qwen3 byte-pair encoding, right-padded to the longest
      prompt (``input_ids`` / ``attention_mask``).

    Args:
        model_type: The diffusion family (see :class:`ZMDiffusion`).
        hf_id: Hosted repo to read ``tokenizer.json`` (and ``tokenizer_3.json``) from.
            Required unless ``tokenizer_file`` is given (no default repo).
        tokenizer_file: Explicit ``tokenizer.json`` path (overrides ``hf_id``).
        tokenizer_file_3: Explicit T5 ``tokenizer.json`` path (SD 3 / 3.5).
        max_seq_len: Padded / truncated length (77 for CLIP; the prompt length
            limit for Qwen, 1024, not counting the template prefix).
        max_sequence_length: T5 padded / truncated length (SD 3 / 3.5, 256).
        pad_token: CLIP pad token string (``<|endoftext|>``, ``!`` for SD 2.x).
    """

    def __init__(
        self,
        model_type=None,
        hf_id=None,
        tokenizer_file=None,
        tokenizer_file_3=None,
        max_seq_len=None,
        max_sequence_length=None,
        pad_token=None,
        **kwargs,
    ):
        spec = DIFFUSION_TOKENIZERS.get(model_type)
        if spec is None:
            raise ValueError(
                f"Unknown diffusion model_type {model_type!r}. ZMDiffusionTokenizer "
                f"supports: {', '.join(sorted(DIFFUSION_TOKENIZERS))}."
            )
        if tokenizer_file is None and hf_id is None:
            raise ValueError(
                "ZMDiffusionTokenizer() needs an hf_id (a hosted repo, read from its "
                "tokenizer.json) or an explicit tokenizer_file: there is no default "
                "repo. Load it by repo id, e.g. "
                "from_weights('zeromodels/stable-diffusion-v1-5')."
            )
        super().__init__(**kwargs)
        self.model_type = model_type
        self.kind = spec["kind"]
        self.hf_id = hf_id
        self.tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        if max_seq_len is None:
            max_seq_len = DEFAULT_MAX_SEQ_LEN[self.kind]
        self.max_seq_len = max_seq_len
        self.pad_token = None
        if self.kind == "clip":
            self.pad_token = pad_token or spec["pad_token"]
            self.tokenizer = CLIPTokenizer(
                tokenizer_file=self.tokenizer_file,
                max_seq_len=max_seq_len,
                pad_token=self.pad_token,
            )
        else:
            self.prompt_template = spec["prompt_template"]
            self.prompt_template_start_idx = spec["prompt_template_start_idx"]
            self.empty_prompt = spec["empty_prompt"]
            self.tokenizer = spec["tokenizer_class"](tokenizer_file=self.tokenizer_file)
        self.tokenizer_file_3 = None
        self.max_sequence_length = None
        self.tokenizer_3 = None
        if spec.get("t5"):
            self.tokenizer_file_3 = self.resolve_tokenizer_json_from_hf(
                hf_id, tokenizer_file_3, filename="tokenizer_3.json"
            )
            self.max_sequence_length = max_sequence_length or 256
            self.tokenizer_3 = T5Tokenizer(
                tokenizer_file=self.tokenizer_file_3,
                max_seq_len=self.max_sequence_length,
            )

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        from zeromodels.conversion.zm_config import (
            ZM_METADATA_KEYS,
            load_zm_config,
            load_zm_preprocessor,
        )

        repo_id = repo_id.rstrip("/")
        spec = load_zm_preprocessor(repo_id)
        if spec is not None:
            config = {k: v for k, v in spec.items() if k not in ZM_METADATA_KEYS}
        else:
            # repos without zm_preprocessor.json: the family comes from the model's
            # zm_config.json, the rest from the family defaults
            model_spec = load_zm_config(repo_id)
            if not model_spec or "model_type" not in model_spec:
                raise OSError(
                    f"{repo_id} has neither zm_preprocessor.json nor a zm_config.json "
                    f"with a model_type, so the diffusion family is unknown. Pass "
                    f"model_type= explicitly."
                )
            config = {"model_type": model_spec["model_type"]}
        config.update(kwargs)
        config.setdefault("hf_id", repo_id)
        return cls(**config)

    @classmethod
    def from_variant(cls, variant, /, **kwargs):
        return cls.from_hub_repo(f"zeromodels/{variant}", **kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        raise NotImplementedError(
            "On-the-fly `hf:` loading is not supported for diffusion models: use "
            "ZMDiffusionTokenizer.from_weights('zeromodels/<variant>')."
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def format_prompt(self, text):
        return self.prompt_template.format(text if text else self.empty_prompt)

    def call(self, inputs):
        if self.kind == "qwen":
            texts = self.tokenizer.normalize_texts(inputs)
            max_length = self.max_seq_len + self.prompt_template_start_idx
            encoded = [
                self.tokenizer.encode(self.format_prompt(t))[:max_length] for t in texts
            ]
            input_ids, attention_mask = self.pad_batch(encoded)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        out = self.tokenizer(inputs)
        if self.tokenizer_3 is not None:
            # the T5 tokenizer truncates to max_sequence_length and pads to the
            # longest prompt; the transformer takes the fixed length, so pad the rest
            ids = self.tokenizer_3(inputs)["input_ids"]
            short = self.max_sequence_length - int(ids.shape[1])
            if short > 0:
                ids = ops.pad(
                    ids,
                    ((0, 0), (0, short)),
                    constant_values=self.tokenizer_3.pad_token_id,
                )
            out["input_ids_3"] = ids
        return out

    def decode(self, ids, skip_special_tokens=True):
        if self.kind == "qwen":
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return self.tokenizer.detokenize(ids)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_type": self.model_type,
                "hf_id": self.hf_id,
                "tokenizer_file": self.tokenizer_file,
                "tokenizer_file_3": self.tokenizer_file_3,
                "max_seq_len": self.max_seq_len,
                "max_sequence_length": self.max_sequence_length,
                "pad_token": self.pad_token,
            }
        )
        return config

    def get_hub_config(self):
        # the portable zm_preprocessor.json recipe: the family and its settings,
        # with the tokenizer files read from the same repo (no local paths)
        config = {"model_type": self.model_type, "max_seq_len": self.max_seq_len}
        if self.pad_token is not None:
            config["pad_token"] = self.pad_token
        if self.max_sequence_length is not None:
            config["max_sequence_length"] = self.max_sequence_length
        return config
