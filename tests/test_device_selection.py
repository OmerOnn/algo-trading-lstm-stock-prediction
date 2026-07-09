import unittest
from unittest.mock import patch

import torch

from src.device import get_best_device


class DeviceSelectionTest(unittest.TestCase):
    def test_mac_alias_selects_mps_when_available(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available",
            return_value=True,
        ), patch("torch.backends.mps.is_built", return_value=True):
            device_info = get_best_device("mac")

        self.assertEqual(device_info.device, torch.device("mps"))
        self.assertEqual(device_info.accelerator, "MPS / Apple Metal GPU")

    def test_mps_request_fails_clearly_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available",
            return_value=False,
        ), patch("torch.backends.mps.is_built", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                get_best_device("metal")


if __name__ == "__main__":
    unittest.main()
