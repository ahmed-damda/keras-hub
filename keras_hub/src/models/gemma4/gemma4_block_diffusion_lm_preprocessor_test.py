import numpy as np
import pytest

from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_preprocessor import (
    Gemma4BlockDiffusionLMPreprocessor,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma4BlockDiffusionLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # Text-only preprocessor (no media converters).
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
            "canvas_length": 4,
        }
        self.preprocessor = Gemma4BlockDiffusionLMPreprocessor(
            **self.init_kwargs
        )

    # ------------------------------------------------------------------
    # Training path (call())
    # ------------------------------------------------------------------

    def test_preprocessor_call_shape(self):
        """call() packs strings into (token_ids, padding_mask) of the
        correct shape."""
        output = self.run_preprocessing_layer_test(
            cls=Gemma4BlockDiffusionLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=["the quick brown fox", "the quick brown fox"],
            return_output=True,
        )
        x, y, sw = output
        self.assertEqual(x["token_ids"].shape[-1], 8)
        self.assertEqual(x["padding_mask"].shape[-1], 8)
        self.assertEqual(y.shape[-1], 8)

    def test_no_start_end_token(self):
        """add_start_token=False/add_end_token=False strips BOS/EOS."""
        preprocessor = Gemma4BlockDiffusionLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        input_data = ["the quick brown fox"] * 2
        x, y, sw = preprocessor(input_data)
        # Without BOS the first token should be "the" (id=9).
        self.assertAllEqual(x["token_ids"][0, 0], 9)

    def test_generate_preprocess_appends_canvas(self):
        """generate_preprocess() extends token_ids by canvas_length mask
        tokens."""
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        prompt_plus_canvas = 8 + 4  # sequence_length + canvas_length
        self.assertAllEqual(output["token_ids"].shape[-1], prompt_plus_canvas)
        self.assertAllEqual(
            output["padding_mask"].shape[-1], prompt_plus_canvas
        )

    def test_generate_preprocess_canvas_mask_is_zero(self):
        """Canvas positions in padding_mask are 0 (not in the prompt)."""
        output = self.preprocessor.generate_preprocess("the quick brown fox")
        padding_mask = np.array(output["padding_mask"])
        # Canvas positions (last canvas_length entries) must all be 0.
        canvas_mask = padding_mask[..., -self.preprocessor.canvas_length :]
        self.assertAllEqual(canvas_mask, np.zeros_like(canvas_mask))

    def test_generate_preprocess_batched(self):
        """generate_preprocess() works with a list of strings."""
        output = self.preprocessor.generate_preprocess(
            {"prompts": ["the quick brown fox", "the quick brown fox"]}
        )
        self.assertEqual(output["token_ids"].shape[0], 2)

    def test_generate_postprocess(self):
        """generate_postprocess() converts canvas token ids back to strings."""
        canvas = np.array([9, 14, 10, 12, 0, 0, 0, 0], dtype="int32")
        result = self.preprocessor.generate_postprocess(canvas)
        self.assertAllEqual(result, "the quick brown fox")

    def test_generate_postprocess_batched(self):
        """generate_postprocess() handles a 2-D canvas batch."""
        canvas = np.array(
            [[9, 14, 10, 12, 0, 0, 0, 0], [9, 14, 10, 12, 0, 0, 0, 0]],
            dtype="int32",
        )
        results = self.preprocessor.generate_postprocess(canvas)
        self.assertEqual(len(results), 2)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_serialization(self):
        """Preprocessor config roundtrip preserves all parameters."""
        self.run_serialization_test(self.preprocessor)

    def test_sequence_length_setter(self):
        """sequence_length setter updates the underlying packer."""
        self.preprocessor.sequence_length = 16
        self.assertEqual(self.preprocessor.sequence_length, 16)

    # ------------------------------------------------------------------
    # Preset smoke-test (requires credentials)
    # ------------------------------------------------------------------

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        for preset in Gemma4BlockDiffusionLMPreprocessor.presets:
            self.run_preset_test(
                cls=Gemma4BlockDiffusionLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )
