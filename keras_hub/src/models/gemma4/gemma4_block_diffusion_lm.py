from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.block_diffusion_lm import BlockDiffusionLM
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_layers import (
    Gemma4BlockDiffusionSelfConditioning,
)
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_preprocessor import (
    Gemma4BlockDiffusionLMPreprocessor,
)


@keras_hub_export("keras_hub.models.Gemma4BlockDiffusionLM")
class Gemma4BlockDiffusionLM(BlockDiffusionLM):
    """Gemma4-based discrete block-diffusion language model.

    Wraps a `Gemma4Backbone` with the block-diffusion generation loop from
    `DiffusionLM`.  The backbone is called twice per generation iteration: once
    as a causal encoder to freeze prompt KV caches, and up to
    `max_denoising_steps` times as a bidirectional decoder over a fixed-length
    canvas of tokens.

    Args:
        backbone: A `keras_hub.models.Gemma4Backbone` instance.
        preprocessor: A `keras_hub.models.Gemma4BlockDiffusionLMPreprocessor`
            or `None`.
    """

    backbone_cls = Gemma4Backbone
    preprocessor_cls = Gemma4BlockDiffusionLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        self.backbone = backbone
        self.preprocessor = preprocessor
        super().__init__(**kwargs)
        self.diffusion_self_conditioning = Gemma4BlockDiffusionSelfConditioning(
            hidden_dim=backbone.hidden_dim,
            intermediate_dim=backbone.intermediate_dim,
            dtype=self.dtype_policy,
            name="diffusion_self_conditioning",
        )

        self.diffusion_self_conditioning.build(
            (None, None, backbone.hidden_dim)
        )

    def _encode_prompt(self, inputs):
        token_ids = inputs["token_ids"]
        padding_mask = inputs.get("padding_mask", None)

        # Multimodal vision inputs (images or video frames).
        pixel_values = inputs.get("pixel_values", None)
        pixel_position_ids = inputs.get("pixel_position_ids", None)
        vision_indices = inputs.get("vision_indices", None)
        vision_mask = inputs.get("vision_mask", None)

        # Text embeddings — kept unscaled until after vision interleaving so
        # that pre-scaled vision embeddings land at the correct magnitude after
        # the global x *= sqrt(hidden_dim) step below.
        x = self.backbone.token_embedding(token_ids)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )

        # Interleave vision embeddings (images or video frames).
        # Pre-scale by 1/sqrt(hidden_dim) so that after the global scale the
        # vision positions stay at their natural embed_vision magnitude,
        # matching the pattern in Gemma4CausalLM.call_with_cache().
        num_images = 0
        if (
            pixel_values is not None
            and hasattr(pixel_values, "shape")
            and len(pixel_values.shape) > 1
        ):
            num_images = pixel_values.shape[1]

        if not self.backbone.text_only_model and num_images:
            img_embeddings = self.backbone.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
            scaled_img_embeddings = img_embeddings * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, img_embeddings.dtype
            )
            x = self.backbone.interleave_embeddings(
                image_embeddings=scaled_img_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices,
            )
            vision_mask = ops.cast(vision_mask, "bool")
        else:
            vision_mask = None

        # Per-layer token embeddings: zero out vision positions so they don't
        # contribute a spurious text-token embedding at those positions.
        _hpl = self.backbone.hidden_size_per_layer_input
        if _hpl > 0:
            _per_layer_ids = token_ids
            if vision_mask is not None:
                _per_layer_ids = ops.where(
                    vision_mask,
                    ops.zeros_like(_per_layer_ids),
                    _per_layer_ids,
                )
            _per_emb = self.backbone.per_layer_token_embedding(_per_layer_ids)
            _per_emb = ops.cast(_per_emb, x.dtype)
            _per_emb = _per_emb * ops.cast(float(_hpl) ** 0.5, _per_emb.dtype)
            per_layer_emb_flat = _per_emb
        else:
            per_layer_emb_flat = None

        # Global scale applied after interleaving: text positions get
        # sqrt(hidden_dim), vision positions keep their pre-scaled magnitude.
        x = x * embed_scale

        if _hpl > 0:
            _per_proj = self.backbone.per_layer_model_projection(x)
            _per_proj = _per_proj * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, _per_proj.dtype
            )
            per_layer_proj_flat = _per_proj
        else:
            per_layer_proj_flat = None

        batch_size = ops.shape(token_ids)[0]
        prompt_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        global_head_dim = self.backbone.global_head_dim
        max_head_dim = (
            max(head_dim, global_head_dim) if global_head_dim else head_dim
        )
        cache_shape = [
            batch_size,
            num_layers,
            2,
            prompt_length,
            num_heads,
            max_head_dim,
        ]
        cache = ops.zeros(cache_shape, dtype=self.compute_dtype)

        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            shared_kv = None
            if (
                layer.is_kv_shared_layer
                and layer.kv_shared_layer_index is not None
            ):
                idx = layer.kv_shared_layer_index
                if idx < len(caches):
                    shared_kv = caches[idx]
                else:
                    shared_kv = cache[:, idx, ...]

            if per_layer_proj_flat is not None:
                proj_i = per_layer_proj_flat[:, :, i * _hpl : (i + 1) * _hpl]
                emb_i = per_layer_emb_flat[:, :, i * _hpl : (i + 1) * _hpl]
                proj_i_normed = self.backbone.per_layer_projection_norm(proj_i)
                per_layer_input_i = (proj_i_normed + emb_i) * ops.cast(
                    2.0**-0.5, proj_i.dtype
                )
            else:
                per_layer_input_i = None

            x, next_cache = layer(
                x,
                cache=current_cache,
                cache_update_index=0,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                shared_kv=shared_kv,
                per_layer_input=per_layer_input_i,
                use_encoder_scalar=True,
            )
            caches.append(next_cache)

        encoder_kv_cache = ops.stack(caches, axis=1)
        return encoder_kv_cache, prompt_length

    def _prepare_canvas_embeds(self, canvas, prev_logits):
        x = self.backbone.token_embedding(canvas)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        x = x + self.diffusion_self_conditioning(
            x,
            prev_logits,
            self.backbone.token_embedding.embeddings,
            embed_scale,
        )
        return x

    def _decode_canvas_step(
        self, canvas_embeds, encoder_kv_cache, prompt_length
    ):
        x = canvas_embeds
        batch_size = ops.shape(x)[0]
        canvas_length = ops.shape(x)[1]

        # Build a combined cache: the encoder slice is pre-filled; the canvas
        # slice will be written during this forward pass.  Only the sequence
        # axis (axis 3) needs padding; all other dims match the encoder cache.

        # Pad encoder cache to cover prompt + canvas along the sequence axis
        # (axis 3).  encoder_kv_cache shape: (B, L, 2, prompt_len, heads, hd)
        pad_len = canvas_length
        # Pad: (before, after) for each dimension — only pad axis 3.
        paddings = [
            [0, 0],  # batch
            [0, 0],  # layers
            [0, 0],  # 2 (K/V)
            [0, pad_len],  # sequence
            [0, 0],  # heads
            [0, 0],  # head_dim
        ]
        combined_cache = ops.pad(encoder_kv_cache, paddings)

        # canvas_mask marks every canvas position as bidirectional.
        canvas_mask = ops.ones((batch_size, canvas_length), dtype="bool")

        _hpl = self.backbone.hidden_size_per_layer_input
        if _hpl > 0:
            _per_emb = self.backbone.per_layer_token_embedding(
                ops.zeros((batch_size, canvas_length), dtype="int32")
            )
            _per_emb = ops.cast(_per_emb, x.dtype)
            _per_emb = _per_emb * ops.cast(float(_hpl) ** 0.5, _per_emb.dtype)
            per_layer_emb_flat = _per_emb

            _per_proj = self.backbone.per_layer_model_projection(x)
            _per_proj = _per_proj * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, _per_proj.dtype
            )
            per_layer_proj_flat = _per_proj
        else:
            per_layer_emb_flat = None
            per_layer_proj_flat = None

        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = combined_cache[:, i, ...]
            shared_kv = None
            if (
                layer.is_kv_shared_layer
                and layer.kv_shared_layer_index is not None
            ):
                idx = layer.kv_shared_layer_index
                if idx < len(caches):
                    shared_kv = caches[idx]
                else:
                    shared_kv = combined_cache[:, idx, ...]

            if per_layer_proj_flat is not None:
                proj_i = per_layer_proj_flat[:, :, i * _hpl : (i + 1) * _hpl]
                emb_i = per_layer_emb_flat[:, :, i * _hpl : (i + 1) * _hpl]
                proj_i_normed = self.backbone.per_layer_projection_norm(proj_i)
                per_layer_input_i = (proj_i_normed + emb_i) * ops.cast(
                    2.0**-0.5, proj_i.dtype
                )
            else:
                per_layer_input_i = None

            x, next_cache = layer(
                x,
                cache=current_cache,
                cache_update_index=prompt_length,
                canvas_mask=canvas_mask,
                shared_kv=shared_kv,
                per_layer_input=per_layer_input_i,
            )
            caches.append(next_cache)

        return self.backbone.layer_norm(x)

    def _canvas_logits(self, hidden):
        logits = self.backbone.token_embedding(hidden, reverse=True)
        soft_cap = self.backbone.final_logit_soft_cap
        if soft_cap is not None:
            logits = ops.tanh(logits / soft_cap) * soft_cap
        return logits

    def call(self, x, training=False):
        token_ids = x["token_ids"]
        padding_mask = x.get("padding_mask", None)

        backbone_inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "position_ids": ops.expand_dims(
                ops.arange(ops.shape(token_ids)[1], dtype="int32"), axis=0
            ),
        }
        # Pass vision fields when the backbone has a vision encoder so that
        # the backbone's functional graph receives all required inputs.
        if not self.backbone.text_only_model:
            for key in (
                "pixel_values",
                "pixel_position_ids",
                "vision_indices",
                "vision_mask",
            ):
                if key in x:
                    backbone_inputs[key] = x[key]
        hidden = self.backbone(backbone_inputs, training=training)
        return self._canvas_logits(hidden)
