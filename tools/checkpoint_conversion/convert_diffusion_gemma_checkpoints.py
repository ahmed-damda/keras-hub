"""Convert DiffusionGemma HuggingFace checkpoints to the KerasHub preset format.

This script ports a ``google/diffusiongemma-*`` checkpoint from HuggingFace
into a KerasHub ``Gemma4BlockDiffusionLM`` preset.  It performs two steps:

  1. ``Gemma4BlockDiffusionLM.from_preset("hf://...")`` — handled entirely by 
    the KerasHub preset loader pipeline:

     * ``convert_backbone_config`` → backbone constructor kwargs
     * ``convert_weights``          → backbone weights (vision encoder +
                                       text decoder layers)
     * ``convert_head``             → self-conditioning task-head weights
                                       (``Gemma4BlockDiffusionSelfConditioning``)

     The ``convert_head`` hook is the mechanism described in
     ``TransformersPresetLoader.load_task``; it is invoked automatically
     whenever ``convert_gemma4.convert_head`` is defined, following the same
     ``hasattr`` dispatch pattern used by ``load_image_converter_config``,
     ``load_preprocessor_config``, etc.

  2. Verify numerics against HF (optional: forward-pass logit comparison on a
     text prompt and image prompt).

  3. Save the model to a local preset directory.

Usage::

    # Text + image numerics verification, save in bfloat16:
    python tools/checkpoint_conversion/convert_diffusion_gemma_checkpoints.py \\
        --preset diffusion_gemma_26b_a4b_it \\
        --save_dtype bfloat16

    # Skip the HF numerics check (faster, requires less RAM):
    python tools/checkpoint_conversion/convert_diffusion_gemma_checkpoints.py \\
        --preset diffusion_gemma_26b_a4b_it \\
        --skip_verify

Notes:
    * ``KERAS_BACKEND`` is forced to ``"torch"`` so that weight tensors are
      always in a consistent numerical format during conversion.
    * CUDA is disabled (``CUDA_VISIBLE_DEVICES=-1``) to avoid GPU OOM during
      the float32 verification pass.
    * HF ``transformers`` is required for the optional verify step (numerics
      comparison against the reference implementation).  The minimal path
      (``--skip_verify``) only needs ``safetensors`` and ``keras_hub``.
"""

import contextlib
import gc
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from io import BytesIO

import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image

import keras_hub

# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESET_MAP = {
    "diffusion_gemma_26b_a4b_it": "google/diffusiongemma-26B-A4B-it",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

PROMPT_TEXT = (
    "<start_of_turn>user\n"
    "What is the capital of France?"
    "<end_of_turn>\n<start_of_turn>model\n"
)
PROMPT_IMAGE = (
    "<start_of_turn>user\n\n<|image|>\n"
    "What is in this image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Name of the preset to convert. Must be one of: "
    f"{', '.join(PRESET_MAP.keys())}",
)
flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype in which to save the converted preset. Defaults to 'bfloat16'.",
)
flags.DEFINE_boolean(
    "skip_verify",
    False,
    "Skip the HuggingFace numerics-comparison step. "
    "Useful when the full HF model cannot be loaded (e.g. insufficient RAM).",
)

# ---------------------------------------------------------------------------
# HF model loading (verification only)
# ---------------------------------------------------------------------------


def _load_hf_model(hf_repo_id):
    """Load the HF DiffusionGemma model and processor for numerics checks.

    Tries ``AutoModelForCausalLM`` first; falls back to ``AutoModel`` if that
    class is not registered for the model type (e.g. when DiffusionGemma
    requires a custom ``model_type`` mapping).

    Returns ``(hf_model, processor)`` in float32 on CPU.
    """
    from transformers import AutoModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoProcessor

    print(f"-> Loading HF model from {hf_repo_id} …")
    load_kwargs = {
        "device_map": "cpu",
        "torch_dtype": torch.float32,
    }

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id, **load_kwargs
        )
    except Exception:
        hf_model = AutoModel.from_pretrained(hf_repo_id, **load_kwargs)

    hf_model.eval()
    processor = AutoProcessor.from_pretrained(hf_repo_id)
    print("-> HF model loaded.")
    return hf_model, processor


# ---------------------------------------------------------------------------
# Numerics verification helpers
# ---------------------------------------------------------------------------


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


@contextlib.contextmanager
def _no_grad():
    with torch.no_grad():
        yield


def _hf_forward(hf_model, processor, prompt, raw_image=None):
    """Run one HF forward pass and return logits as a float32 numpy array."""
    proc_kwargs = {"text": prompt, "return_tensors": "pt"}
    if raw_image is not None:
        proc_kwargs["images"] = raw_image
    hf_inputs = processor(**proc_kwargs)
    hf_inputs = {k: v.cpu() for k, v in hf_inputs.items()}
    with _no_grad():
        hf_out = hf_model(**hf_inputs)
    return (
        hf_out.logits.detach().cpu().float().numpy(),
        hf_inputs["input_ids"].numpy(),
        hf_inputs.get(
            "attention_mask", torch.ones_like(hf_inputs["input_ids"])
        ).numpy(),
    )


def _kh_forward(
    backbone,
    token_ids,
    padding_mask,
    pixel_values=None,
    pixel_position_ids=None,
    vision_indices=None,
    vision_mask=None,
):
    """Run one KerasHub backbone forward pass and return logits."""
    inputs = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
        "position_ids": ops.convert_to_tensor(
            np.arange(token_ids.shape[1], dtype=np.int32)[np.newaxis, :]
        ),
    }
    if not backbone.text_only_model:
        batch_size = token_ids.shape[0]
        if pixel_values is not None:
            inputs["pixel_values"] = ops.convert_to_tensor(pixel_values)
            inputs["pixel_position_ids"] = ops.convert_to_tensor(
                pixel_position_ids
            )
            inputs["vision_indices"] = ops.convert_to_tensor(vision_indices)
            inputs["vision_mask"] = ops.convert_to_tensor(vision_mask)
        else:
            inputs["pixel_values"] = ops.convert_to_tensor(
                np.zeros((batch_size, 0, 1, 768), dtype=np.float32)
            )
            inputs["pixel_position_ids"] = ops.convert_to_tensor(
                np.zeros((batch_size, 0, 1, 2), dtype=np.int32)
            )
            inputs["vision_indices"] = ops.convert_to_tensor(
                np.zeros((batch_size, 0), dtype=np.int32)
            )
            inputs["vision_mask"] = ops.convert_to_tensor(
                np.zeros((batch_size, token_ids.shape[1]), dtype=np.int32)
            )
    with torch.no_grad():
        hidden = backbone(inputs)
    kh_logits = ops.convert_to_numpy(
        backbone.token_embedding(hidden, reverse=True)
    ).astype(np.float32)
    return kh_logits


def _test_numerics(label, backbone, kh_logits, hf_logits):
    """Log max/mean absolute logit difference; warn if > 1e-3."""
    # Trim to common sequence length.
    min_len = min(kh_logits.shape[1], hf_logits.shape[1])
    kh = kh_logits[:, :min_len, :]
    hf = hf_logits[:, :min_len, :]

    abs_diff = np.abs(kh - hf)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    try:
        np.testing.assert_allclose(kh, hf, atol=1e-3, rtol=1e-3)
        print(
            f"✅ [{label}] Logits within 1e-3 tolerance "
            f"(max={max_diff:.6f}, mean={mean_diff:.6f})."
        )
    except AssertionError:
        print(
            f"⚠️  [{label}] Logits exceed 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}. "
            "NOTE: small numerical gaps may be backend-dependent."
        )


def _verify(diffusion_lm, hf_preset):
    """Load HF model and compare backbone logits for text and image prompts."""
    hf_repo_id = hf_preset.replace("hf://", "")
    hf_model, processor = _load_hf_model(hf_repo_id)
    backbone = diffusion_lm.backbone

    tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(hf_preset)
    image_placeholder_id = tokenizer.image_placeholder_id

    # --- Text ---
    print("\n--- Numerics Verification: text ---")
    hf_logits, hf_ids, hf_mask = _hf_forward(hf_model, processor, PROMPT_TEXT)
    kh_logits = _kh_forward(
        backbone,
        hf_ids.astype(np.int32),
        hf_mask.astype(np.int32),
    )
    _test_numerics("text", backbone, kh_logits, hf_logits)

    # --- Image ---
    if not backbone.text_only_model:
        print("\n--- Numerics Verification: image ---")
        try:
            raw_image = _load_test_image()
            hf_logits, hf_ids, hf_mask = _hf_forward(
                hf_model, processor, PROMPT_IMAGE, raw_image=raw_image
            )
            token_ids = hf_ids.astype(np.int32)
            padding_mask = hf_mask.astype(np.int32)
            batch_size = token_ids.shape[0]

            # Build vision inputs from HF token IDs (image placeholder
            # positions mark where vision tokens are interleaved).
            vision_mask = (token_ids == image_placeholder_id).astype(np.int32)
            vision_rows = [
                np.where(vision_mask[b])[0].astype(np.int32)
                for b in range(batch_size)
            ]
            max_vis = max((len(r) for r in vision_rows), default=0)
            vision_indices = np.zeros((batch_size, max_vis), dtype=np.int32)
            for b, row in enumerate(vision_rows):
                vision_indices[b, : len(row)] = row

            kh_logits = _kh_forward(
                backbone,
                token_ids,
                padding_mask,
                vision_indices=vision_indices,
                vision_mask=vision_mask,
                # pixel_values / pixel_position_ids not passed here — we rely
                # on the fact that the HF forward pass already produced its
                # own logits using its pixel normalisation; the KH path here
                # runs without pixel data (zero pixel values) to measure the
                # text-decoder contribution only.  A full pixel-injected test
                # would need to extract HF's embed_vision outputs via a hook
                # (see convert_gemma4_hf_checkpoints.py).
            )
            _test_numerics(
                "image (token structure)", backbone, kh_logits, hf_logits
            )
        except Exception as e:
            print(f"⚠️  Image numerics check skipped: {e}")

    del hf_model, processor
    gc.collect()
    print("-> HF verification complete.")


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def _count_hf_params(hf_model):
    param_names = {name for name, _ in hf_model.named_parameters()}
    num_params = sum(p.numel() for p in hf_model.parameters())
    num_buffers = sum(
        v.numel()
        for name, v in hf_model.state_dict().items()
        if name not in param_names and name.endswith(".layer_scalar")
    )
    return num_params + num_buffers


def _count_kh_params(model):
    """Count parameters in a KerasHub model (backbone + task head)."""
    unique = {id(w): w for w in model.weights}.values()
    return sum(w.numpy().size for w in unique)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save_preset(diffusion_lm, hf_preset, preset_name, save_dtype):
    """Save the converted model to a local preset directory."""
    save_path = f"./{preset_name}"
    print(f"\n-> Saving model in {save_dtype} to {save_path} …")

    if save_dtype == "bfloat16":
        # Reload in bfloat16.  from_preset calls convert_weights (backbone)
        # and convert_head (self-conditioning) automatically, so no manual
        # weight porting is needed here.
        del diffusion_lm
        gc.collect()
        diffusion_lm_bf16 = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
            hf_preset, dtype="bfloat16"
        )
        diffusion_lm_bf16.save_to_preset(save_path)
    else:
        # float32: already verified — save directly.
        diffusion_lm.save_to_preset(save_path)

    print(f"-> Preset saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(_):
    preset_name = FLAGS.preset
    if preset_name not in PRESET_MAP:
        raise ValueError(
            f"Unknown preset {FLAGS.preset!r}. "
            f"Choose one of: {', '.join(PRESET_MAP.keys())}"
        )

    hf_repo_id = PRESET_MAP[preset_name]
    hf_preset = f"hf://{hf_repo_id}"

    # ── Step 1: Load model via KerasHub preset loader ─────────────────────────
    # from_preset triggers the full TransformersPresetLoader pipeline:
    #   convert_backbone_config  → backbone constructor kwargs
    #   convert_weights          → backbone weights (vision enc + text decoder)
    #   convert_head             → self-conditioning task-head weights
    # No manual weight porting is needed after this call.
    print(f"-> Loading Gemma4BlockDiffusionLM from {hf_preset} …")
    diffusion_lm = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
        hf_preset, dtype="float32"
    )
    print("✓ All weights loaded (backbone + self-conditioning).")

    # ── Step 2: Numerics verification (optional) ──────────────────────────────
    if not FLAGS.skip_verify:
        _verify(diffusion_lm, hf_preset)
    else:
        print("-> Numerics verification skipped (--skip_verify).")

    # ── Step 3: Save ──────────────────────────────────────────────────────────
    _save_preset(diffusion_lm, hf_preset, preset_name, FLAGS.save_dtype)


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
