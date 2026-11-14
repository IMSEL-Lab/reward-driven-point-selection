from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from sam_rl.cli import eval_point_baselines as evaluator


class MaskAwareGreedyTests(unittest.TestCase):
    def test_first_click_is_top_similarity_and_prefix_is_causal(self) -> None:
        sim_map = np.array([[0.6, 0.8, 0.9]], dtype=np.float32)
        observed_prefixes: list[np.ndarray] = []
        predicted = np.zeros((5, 5), dtype=bool)

        def predict(_predictor, prefix: np.ndarray) -> np.ndarray:
            observed_prefixes.append(prefix.copy())
            return predicted

        with patch.object(evaluator, "predict_mask_from_points", side_effect=predict):
            points = evaluator._mask_aware_greedy_points(
                sim_map, count=2, predictor=object(), image_shape=(5, 5), grid=3
            )

        np.testing.assert_array_equal(points[0], [2, 0])
        self.assertEqual(len(observed_prefixes), 1)
        expected_prefix = evaluator._grid_coordinates_to_image(
            np.array([[2, 0]], dtype=np.int32), 3, 5, 5
        )
        np.testing.assert_array_equal(observed_prefixes[0], expected_prefix)

    def test_foreground_check_uses_the_exact_rounded_prompt_pixel(self) -> None:
        sim_map = np.array([[0.6, 0.8, 0.9]], dtype=np.float32)
        predicted = np.zeros((5, 5), dtype=bool)
        # Grid cell (0, 0) maps to pixel (1, 1) after rounding, while flooring
        # the unrounded center would check (0, 0).
        predicted[1, 1] = True

        with patch.object(evaluator, "predict_mask_from_points", return_value=predicted):
            points = evaluator._mask_aware_greedy_points(
                sim_map, count=2, predictor=object(), image_shape=(5, 5), grid=3
            )

        np.testing.assert_array_equal(points, [[2, 0], [1, 0]])

    def test_fully_foreground_mask_falls_back_deterministically_to_unused_cells(self) -> None:
        sim_map = np.array([[0.6, 0.8, 0.9]], dtype=np.float32)
        predicted = np.ones((5, 5), dtype=bool)

        with patch.object(evaluator, "predict_mask_from_points", return_value=predicted):
            first = evaluator._mask_aware_greedy_points(
                sim_map, count=3, predictor=object(), image_shape=(5, 5), grid=3
            )
            second = evaluator._mask_aware_greedy_points(
                sim_map, count=3, predictor=object(), image_shape=(5, 5), grid=3
            )

        np.testing.assert_array_equal(first, [[2, 0], [1, 0], [0, 0]])
        np.testing.assert_array_equal(second, first)
        self.assertEqual(len({tuple(point) for point in first.tolist()}), 3)

    def test_selector_has_no_ground_truth_input(self) -> None:
        parameters = inspect.signature(evaluator._mask_aware_greedy_points).parameters
        self.assertFalse({"query_mask", "target_mask", "ground_truth"}.intersection(parameters))


if __name__ == "__main__":
    unittest.main()
