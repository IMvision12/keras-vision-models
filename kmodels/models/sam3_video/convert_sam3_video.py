"""Convert HuggingFace Sam3VideoModel tracker_neck weights to Keras."""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.sam3_video.modeling_sam3_video import (  # noqa: E402
    Sam3VideoModel as HFSam3VideoModel,
)

from kmodels.models.sam3.sam3_model import Sam3  # noqa: E402
from kmodels.models.sam3_tracker_video.sam3_tracker_video_model import (  # noqa: E402
    Sam3TrackerVideo,
)
from kmodels.models.sam3_video.sam3_video_model import (  # noqa: E402
    Sam3Video,
)
from kmodels.utils.weight_transfer_torch_to_keras import transfer_weights  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN")


def _transfer_conv(keras_conv, hf_weight, hf_bias):
    transfer_weights("conv_kernel", keras_conv.kernel, hf_weight)
    keras_conv.bias.assign(hf_bias)


def _transfer_conv_transpose(keras_conv, hf_weight, hf_bias):
    """Transfer ConvTranspose2d weights."""
    transfer_weights("conv_transpose_kernel", keras_conv.kernel, hf_weight)
    keras_conv.bias.assign(hf_bias)


def convert():
    print("Loading HF Sam3VideoModel...")
    hf_model = HFSam3VideoModel.from_pretrained(
        "facebook/sam3", attn_implementation="eager", token=HF_TOKEN
    ).eval()
    hf = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    print(f"HF model has {len(hf)} weight tensors")

    # Find tracker_neck keys
    neck_keys = [k for k in hf if k.startswith("tracker_neck.")]
    print(f"Tracker neck weights: {len(neck_keys)}")
    for k in neck_keys:
        print(f"  {k}: {hf[k].shape}")

    print("\nCreating Keras models...")
    sam3 = Sam3(input_shape=(1008, 1008, 3), weights=None)
    sam3.load_weights("sam3.weights.h5")

    tracker_video = Sam3TrackerVideo(sam3_model=sam3, weights=None)
    tracker_video.load_weights("sam3_tracker_video.weights.h5")

    video_model = Sam3Video(
        sam3_model=sam3,
        tracker_video_model=tracker_video,
        weights=None,
    )

    # ── Transfer tracker_neck FPN weights ──
    print("Transferring tracker_neck...")
    neck = video_model.tracker_neck

    # FPN layers (4 scales: 4.0, 2.0, 1.0, 0.5)
    scale_factors = [4.0, 2.0, 1.0, 0.5]
    for i, sf in enumerate(scale_factors):
        fpn = neck.fpn_layers[i]
        fp = f"tracker_neck.fpn_layers.{i}"

        if sf == 4.0:
            # scale_layers_0 = ConvTranspose2d(1024→512, k2s2)
            _transfer_conv_transpose(
                fpn._deconv1,
                hf[f"{fp}.scale_layers.0.weight"],
                hf[f"{fp}.scale_layers.0.bias"],
            )
            # scale_layers_2 = ConvTranspose2d(512→256, k2s2)
            _transfer_conv_transpose(
                fpn._deconv2,
                hf[f"{fp}.scale_layers.2.weight"],
                hf[f"{fp}.scale_layers.2.bias"],
            )
        elif sf == 2.0:
            _transfer_conv_transpose(
                fpn._deconv1,
                hf[f"{fp}.scale_layers.0.weight"],
                hf[f"{fp}.scale_layers.0.bias"],
            )
        elif sf == 0.5:
            pass  # MaxPool has no weights

        # proj1 and proj2
        _transfer_conv(fpn.proj1, hf[f"{fp}.proj1.weight"], hf[f"{fp}.proj1.bias"])
        _transfer_conv(fpn.proj2, hf[f"{fp}.proj2.weight"], hf[f"{fp}.proj2.bias"])

    print(
        f"\nVideo model tracker_neck params: {sum(w.numpy().size for w in neck.weights):,}"
    )

    # ── Equivalence test: compare FPN outputs ──
    print("\nRunning FPN neck equivalence test...")
    import urllib.request

    from PIL import Image

    urllib.request.urlretrieve(
        "http://images.cocodataset.org/val2017/000000039769.jpg", "test_image.jpg"
    )
    img = Image.open("test_image.jpg").convert("RGB")

    from kmodels.models.sam3.sam3_processor import preprocess_image

    pixel_values_keras, _ = preprocess_image(img)

    # Get backbone features from detector
    backbone_nchw = video_model.get_backbone_features(pixel_values_keras)
    backbone_np = (
        backbone_nchw.detach().cpu().numpy()
        if hasattr(backbone_nchw, "detach")
        else np.array(backbone_nchw)
    )

    # Keras FPN
    with torch.no_grad():
        fpn_out_k, fpn_pe_k = neck(backbone_nchw)

    # HF FPN (use the same backbone input)
    backbone_torch = torch.from_numpy(backbone_np)
    with torch.no_grad():
        hf_neck = hf_model.tracker_neck
        fpn_out_hf, fpn_pe_hf = hf_neck(backbone_torch)

    from keras import ops

    for i in range(4):
        k_arr = ops.convert_to_numpy(fpn_out_k[i])
        h_arr = fpn_out_hf[i].cpu().numpy()
        diff = np.max(np.abs(k_arr - h_arr))
        print(f"  FPN level {i} diff: {diff:.6e} shape: {k_arr.shape}")

    # Save full video model weights (includes tracker_neck)
    video_model.save_weights("sam3_video.weights.h5")
    print("\nVideo model weights saved: sam3_video.weights.h5")


if __name__ == "__main__":
    convert()
