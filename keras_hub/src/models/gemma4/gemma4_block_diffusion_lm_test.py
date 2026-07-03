import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm import (
    Gemma4BlockDiffusionLM,
)
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_preprocessor import (
    Gemma4BlockDiffusionLMPreprocessor,
)
from keras_hub.src.samplers.entropy_bound_sampler import EntropyBoundSampler
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma4BlockDiffusionLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()
        vocab_size = self.tokenizer.vocabulary_size()

        self.preprocessor = Gemma4BlockDiffusionLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=8,
            canvas_length=4,
        )

        backbone_kwargs = {
            "vocabulary_size": vocab_size,
            "image_size": 16,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "attention_logit_soft_cap": None,
            "final_logit_soft_cap": None,
            "vision_encoder": None,
        }
        self.backbone = Gemma4Backbone(**backbone_kwargs)
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }
        self.sampler = EntropyBoundSampler(vocabulary_size=vocab_size)

        # Pre-processed training inputs for call() tests.
        raw_preprocessed = self.preprocessor(
            ["the quick brown fox", "the quick brown fox"]
        )
        self.input_data = raw_preprocessed[0]

    def test_call_shape(self):
        """call() returns logits with the correct shape."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        logits = model(self.input_data)
        # (batch=2, seq_len=8, vocab_size)
        self.assertEqual(logits.shape, (2, 8, self.tokenizer.vocabulary_size()))

    def test_generate_single_string(self):
        """generate() with a single string prompt returns a string."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        output = model.generate("the quick brown fox")
        self.assertIsInstance(output, str)

    def test_generate_batched_strings(self):
        """generate() with a list of prompts returns a list of strings."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        outputs = model.generate(["the quick brown fox", "the quick brown fox"])
        self.assertEqual(len(outputs), 2)
        for out in outputs:
            self.assertIsInstance(out, str)

    def test_generate_without_preprocessor(self):
        """generate() with preprocessor=None returns raw int canvas."""
        model = Gemma4BlockDiffusionLM(
            backbone=self.backbone,
            preprocessor=None,
            canvas_length=self.preprocessor.canvas_length,
        )
        model.compile(sampler=self.sampler)
        processed = self.preprocessor.generate_preprocess("the quick brown fox")
        # Add batch dimension.
        from keras import ops

        inputs = {
            "token_ids": ops.expand_dims(processed["token_ids"], axis=0),
            "padding_mask": ops.expand_dims(processed["padding_mask"], axis=0),
        }
        output = model.generate(inputs)
        canvas = np.array(output)
        # Shape: (1, canvas_length) or (canvas_length,) after scalar squeeze.
        self.assertEqual(canvas.shape[-1], self.preprocessor.canvas_length)

    def test_generate_compilation_is_cached(self):
        """generate_function is reused across generate() calls."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        model.generate("the quick brown fox")
        first_fn = model.generate_function
        model.generate("the quick brown fox")
        second_fn = model.generate_function
        self.assertEqual(first_fn, second_fn)

    def test_compile_resets_generate_function(self):
        """compile() resets the cached generate_function."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model.compile(sampler=self.sampler)
        model.generate("the quick brown fox")
        model.compile(sampler=self.sampler)
        self.assertIsNone(model.generate_function)

    @parameterized.named_parameters(
        ("default_canvas", {}),
        ("custom_canvas_length", {"canvas_length": 8}),
    )
    def test_serialization(self, extra_kwargs):
        """get_config / from_config roundtrip preserves all parameters."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs, **extra_kwargs)
        self.run_serialization_test(model)

    def test_saved_model(self):
        """Saving and loading weights preserves model outputs."""
        model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        model_output = model(self.input_data)

        path = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save_weights(path)

        restored_model = Gemma4BlockDiffusionLM(**self.init_kwargs)
        # Build the restored model before loading weights.
        _ = restored_model(self.input_data)
        restored_model.load_weights(path)

        # Verify outputs match after weight restore.
        restored_output = restored_model(self.input_data)
        self.assertAllClose(model_output, restored_output, atol=1e-5, rtol=1e-5)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4BlockDiffusionLM.presets:
            self.run_preset_test(
                cls=Gemma4BlockDiffusionLM,
                preset=preset,
                input_data=self.input_data,
            )
