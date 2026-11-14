import unittest
from unittest.mock import patch

from sam_rl.cli.common import choose_device


class DeviceSelectionTests(unittest.TestCase):
    def test_missing_cuda_cannot_silently_start_cpu_training(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                choose_device("cuda:0")
            self.assertEqual(choose_device("cpu"), "cpu")
            self.assertEqual(choose_device("cuda", allow_fallback=True), "cpu")

    def test_gpu_index_is_relative_to_visible_devices(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            self.assertEqual(choose_device("cuda:0"), "cuda:0")
            with self.assertRaises(ValueError):
                choose_device("cuda:1")


if __name__ == "__main__":
    unittest.main()
