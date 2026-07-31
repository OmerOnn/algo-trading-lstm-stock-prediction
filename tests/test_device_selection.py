import unittest
from unittest.mock import patch

import torch

from src.device import get_best_device, resolve_xgboost_backend


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


class XGBoostBackendTest(unittest.TestCase):
    """XGBoost has no Metal backend; these pin the resolution rules that follow."""

    def test_apple_aliases_resolve_to_cpu_rather_than_raising(self):
        # Asking for the Mac GPU is reasonable, so it must degrade with an
        # explanation. Passing "mps" through to XGBoost would raise XGBoostError.
        for alias in ("mps", "metal", "mac", "apple"):
            with self.subTest(alias=alias):
                backend = resolve_xgboost_backend(alias)
                self.assertEqual(backend.device, "cpu")
                self.assertIn("Metal", backend.message)

    def test_only_cpu_and_cuda_are_ever_emitted(self):
        # The device string reaches XGBoost verbatim, and it accepts nothing else.
        for requested in ("auto", "cpu", "mps", "metal", "mac", "apple"):
            with self.subTest(requested=requested):
                self.assertIn(resolve_xgboost_backend(requested).device, {"cpu", "cuda"})

    def test_cuda_needs_a_cuda_compiled_build_not_just_a_gpu(self):
        # device="cuda" on a CPU-only wheel does not raise inside XGBoost, it
        # warns and silently trains on CPU, so a visible GPU alone must not be
        # enough to claim acceleration.
        with patch("src.device._xgboost_build_facts", return_value=("2.1.4", False, True)), patch(
            "torch.cuda.is_available", return_value=True
        ):
            self.assertEqual(resolve_xgboost_backend("auto").device, "cpu")
            with self.assertRaisesRegex(RuntimeError, "compiled without CUDA"):
                resolve_xgboost_backend("cuda")

    def test_cuda_selected_when_build_and_device_both_present(self):
        with patch("src.device._xgboost_build_facts", return_value=("2.1.4", True, True)), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.get_device_name", return_value="Tesla T4"):
            backend = resolve_xgboost_backend("auto")
        self.assertEqual(backend.device, "cuda")
        self.assertEqual(backend.accelerator, "CUDA / NVIDIA GPU")

    def test_cuda_build_without_a_gpu_still_falls_back(self):
        with patch("src.device._xgboost_build_facts", return_value=("2.1.4", True, True)), patch(
            "torch.cuda.is_available", return_value=False
        ):
            self.assertEqual(resolve_xgboost_backend("auto").device, "cpu")
            with self.assertRaisesRegex(RuntimeError, "no CUDA-capable NVIDIA GPU"):
                resolve_xgboost_backend("cuda")

    def test_invalid_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid XGBoost device"):
            resolve_xgboost_backend("opencl")

    def test_reported_backend_is_serialisable_for_run_metadata(self):
        payload = resolve_xgboost_backend("auto").to_dict()
        self.assertEqual(set(payload) >= {"device", "accelerator", "compiled_with_cuda"}, True)
        self.assertIsInstance(payload["compiled_with_cuda"], bool)


class XGBoostDeviceStringTest(unittest.TestCase):
    """Guards the claim above against the installed xgboost, not a mock."""

    def test_installed_xgboost_rejects_metal_device_strings(self):
        xgboost = __import__("importlib").util.find_spec("xgboost")
        if xgboost is None:
            self.skipTest("xgboost is not installed")
        import numpy as np
        from xgboost import XGBRegressor
        from xgboost.core import XGBoostError

        x, y = np.random.default_rng(0).random((32, 3)), np.random.default_rng(1).random(32)
        for rejected in ("mps", "metal"):
            with self.subTest(device=rejected):
                with self.assertRaises(XGBoostError):
                    XGBRegressor(n_estimators=2, device=rejected).fit(x, y)
        XGBRegressor(n_estimators=2, device="cpu").fit(x, y)  # the fallback works


if __name__ == "__main__":
    unittest.main()
