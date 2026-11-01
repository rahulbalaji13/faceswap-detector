"""
Comprehensive test suite for the LFAM (Latent Frequency Attention Module) integration.

Run with:
    python -m pytest test_lfam.py -v
or for a quick smoke test:
    python test_lfam.py

All tests use SYNTHETIC data (random tensors / random face images) so NO
trained checkpoint and NO dataset are required.
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    BRANCH_NAMES,
    BRANCHES,
    LFAM,
    FaceSwapNet,
    fft_magnitude,
    load_checkpoint,
    model_label,
    model_tag,
)
from inference import Detector, to_tensor, topk_mean, is_uncertain
from face_utils import detect_box


# ===========================================================================
# Helper utilities
# ===========================================================================

def _rand_face_tensor(batch=1, channels=3):
    """Random normalized face tensor (B, C, 224, 224)."""
    return torch.randn(batch, channels, 224, 224)


def _rand_bgr_face(size=224):
    """Random BGR uint8 image simulating a face crop."""
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def _make_model(branch="fft+srm", use_lfam=True):
    """Build a FaceSwapNet with pretrained=False (fast, no download)."""
    return FaceSwapNet(branch=branch, pretrained=False, use_lfam=use_lfam).eval()


def _save_checkpoint(model, path, branch="fft+srm", use_lfam=True, threshold=0.5):
    """Save a minimal checkpoint for load_checkpoint() testing."""
    torch.save({
        "model": model.state_dict(),
        "branch": branch,
        "use_freq": "fft" in branch,
        "use_lfam": use_lfam,
        "synthetic_ratio": 0.0,
        "threshold": threshold,
    }, path)


# ===========================================================================
# 1. LFAM unit tests
# ===========================================================================

class TestLFAM:
    """Unit tests for the LFAM module in isolation."""

    def test_output_shape_two_branches(self):
        """Fusing two branches returns (B, HIDDEN) fused and (B, 2) weights."""
        lfam = LFAM([1280, 128]).eval()
        f1 = torch.randn(4, 1280)
        f2 = torch.randn(4, 128)
        fused, weights = lfam(f1, f2)
        assert fused.shape == (4, LFAM.HIDDEN), f"Expected (4, {LFAM.HIDDEN}), got {fused.shape}"
        assert weights.shape == (4, 2), f"Expected (4, 2) weights, got {weights.shape}"

    def test_output_shape_three_branches(self):
        """Fusing three branches returns (B, HIDDEN) fused and (B, 3) weights."""
        lfam = LFAM([1280, 128, 128]).eval()
        f1 = torch.randn(4, 1280)
        f2 = torch.randn(4, 128)
        f3 = torch.randn(4, 128)
        fused, weights = lfam(f1, f2, f3)
        assert fused.shape == (4, LFAM.HIDDEN)
        assert weights.shape == (4, 3)

    def test_weights_sum_to_one(self):
        """Branch attention weights must form a valid probability distribution."""
        lfam = LFAM([1280, 128, 128]).eval()
        f1 = torch.randn(8, 1280)
        f2 = torch.randn(8, 128)
        f3 = torch.randn(8, 128)
        _, weights = lfam(f1, f2, f3)
        row_sums = weights.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5), \
            f"Weights should sum to 1, got: {row_sums}"

    def test_weights_non_negative(self):
        """Softmax weights must all be ≥ 0."""
        lfam = LFAM([1280, 128]).eval()
        _, weights = lfam(torch.randn(4, 1280), torch.randn(4, 128))
        assert (weights >= 0).all(), "All weights should be non-negative"

    def test_temperature_sharpens_routing(self):
        """Lower temperature → more decisive (higher max) weights."""
        torch.manual_seed(42)
        f1 = torch.randn(1, 1280)
        f2 = torch.randn(1, 128)

        lfam_sharp = LFAM([1280, 128], temperature=0.1).eval()
        lfam_flat  = LFAM([1280, 128], temperature=5.0).eval()

        # Copy identical weights so the difference is purely from temperature
        lfam_flat.load_state_dict(lfam_sharp.state_dict())

        with torch.no_grad():
            _, w_sharp = lfam_sharp(f1, f2)
            _, w_flat  = lfam_flat(f1, f2)

        max_sharp = w_sharp.max().item()
        max_flat  = w_flat.max().item()
        assert max_sharp > max_flat, \
            f"Lower temperature should give sharper routing. Got max sharp={max_sharp:.3f}, flat={max_flat:.3f}"

    def test_last_weights_stored(self):
        """LFAM should store last weights as a detached tensor."""
        lfam = LFAM([1280, 128]).eval()
        _ = lfam(torch.randn(2, 1280), torch.randn(2, 128))
        assert lfam._last_weights is not None
        assert lfam._last_weights.shape == (2, 2)
        assert not lfam._last_weights.requires_grad, "Stored weights should be detached"

    def test_gradient_flows_through_lfam(self):
        """Gradients must flow back through LFAM to all branch inputs."""
        lfam = LFAM([128, 64]).train()
        f1 = torch.randn(2, 128, requires_grad=True)
        f2 = torch.randn(2, 64,  requires_grad=True)
        fused, _ = lfam(f1, f2)
        loss = fused.sum()
        loss.backward()
        assert f1.grad is not None, "Gradient should flow back to branch 1"
        assert f2.grad is not None, "Gradient should flow back to branch 2"

    def test_single_branch_not_needed(self):
        """LFAM with a single branch still runs (degenerate case: weight = 1.0 always)."""
        lfam = LFAM([1280]).eval()
        fused, weights = lfam(torch.randn(3, 1280))
        assert fused.shape == (3, LFAM.HIDDEN)
        assert torch.allclose(weights, torch.ones(3, 1), atol=1e-5)

    def test_batch_independence(self):
        """Each sample in a batch should get its own independent weights."""
        lfam = LFAM([1280, 128]).eval()
        # Same feature for all samples → weights should be identical across batch
        f1 = torch.ones(4, 1280)
        f2 = torch.ones(4, 128)
        _, weights = lfam(f1, f2)
        # All rows should be the same (same input → same output)
        diffs = weights - weights[0]
        assert torch.allclose(diffs, torch.zeros_like(diffs), atol=1e-5)


# ===========================================================================
# 2. FaceSwapNet integration tests
# ===========================================================================

class TestFaceSwapNet:
    """Integration tests for FaceSwapNet with and without LFAM."""

    def test_spatial_only_no_lfam(self):
        """spatial branch never uses LFAM (only one branch)."""
        model = _make_model("spatial", use_lfam=True)   # use_lfam=True is a no-op for single branch
        assert not model.use_lfam, "Single-branch model must not have LFAM"
        assert not hasattr(model, "lfam")
        x = _rand_face_tensor()
        out = model(x)
        assert out.shape == (1, 1)

    def test_fft_branch_with_lfam(self):
        """fft branch (spatial + FFT): LFAM active, 2 branch weights."""
        model = _make_model("fft", use_lfam=True)
        assert model.use_lfam
        x = _rand_face_tensor()
        out = model(x)
        assert out.shape == (1, 1)
        assert model._last_weights is not None
        assert model._last_weights.shape == (1, 2)

    def test_srm_branch_with_lfam(self):
        """srm branch (spatial + SRM): LFAM active, 2 branch weights."""
        model = _make_model("srm", use_lfam=True)
        assert model.use_lfam
        x = _rand_face_tensor()
        out = model(x)
        assert out.shape == (1, 1)
        assert model._last_weights.shape == (1, 2)

    def test_fft_srm_branch_with_lfam(self):
        """fft+srm branch: LFAM active, 3 branch weights (spatial + FFT + SRM)."""
        model = _make_model("fft+srm", use_lfam=True)
        assert model.use_lfam
        x = _rand_face_tensor()
        out = model(x)
        assert out.shape == (1, 1)
        assert model._last_weights.shape == (1, 3)

    def test_fft_branch_without_lfam(self):
        """Disabling LFAM falls back to original concatenation."""
        model = _make_model("fft", use_lfam=False)
        assert not model.use_lfam
        assert not hasattr(model, "lfam")
        x = _rand_face_tensor()
        out = model(x)
        assert out.shape == (1, 1)
        assert model._last_weights is None

    def test_lfam_weights_sum_to_one_end_to_end(self):
        """Branch weights stored on model must sum to 1 for every sample."""
        model = _make_model("fft+srm", use_lfam=True)
        x = _rand_face_tensor(batch=6)
        model(x)
        weights = model._last_weights           # (6, 3)
        row_sums = weights.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(6), atol=1e-5)

    def test_dual_view_six_channel_input(self):
        """6-channel dual-view input (training mode) must work correctly."""
        model = _make_model("fft+srm", use_lfam=True)
        x = _rand_face_tensor(batch=2, channels=6)
        out = model(x)
        assert out.shape == (2, 1)

    def test_output_logit_is_scalar_per_sample(self):
        """Output should be a single logit per sample (before sigmoid)."""
        model = _make_model("fft+srm", use_lfam=True)
        x = _rand_face_tensor(batch=5)
        out = model(x)
        assert out.shape == (5, 1)
        probs = torch.sigmoid(out)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_eval_mode_no_dropout_effect(self):
        """In eval mode, two identical forward passes must give identical results."""
        model = _make_model("fft+srm", use_lfam=True)
        x = _rand_face_tensor(batch=4)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Eval mode results must be deterministic"

    def test_gradient_flows_through_full_model(self):
        """Backward pass through the complete model (needed for Grad-CAM)."""
        model = _make_model("fft+srm", use_lfam=True).train()
        x = _rand_face_tensor(batch=2)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that spatial backbone has gradients
        for p in model.spatial.parameters():
            if p.requires_grad and p.grad is not None:
                break
        else:
            pytest.fail("No gradient found in spatial branch parameters")

    def test_different_branches_different_param_counts(self):
        """Verify LFAM adds parameters over the no-lfam baseline."""
        m_with    = _make_model("fft+srm", use_lfam=True)
        m_without = _make_model("fft+srm", use_lfam=False)
        n_with    = sum(p.numel() for p in m_with.parameters())
        n_without = sum(p.numel() for p in m_without.parameters())
        print(f"\n  Params with LFAM:    {n_with:,}")
        print(f"  Params without LFAM: {n_without:,}")
        print(f"  LFAM overhead:       {n_with - n_without:,}")
        assert n_with > n_without, "LFAM model should have more parameters"


# ===========================================================================
# 3. Checkpoint save / load tests
# ===========================================================================

class TestCheckpoint:
    """Tests for checkpoint serialization and deserialization."""

    def test_save_and_load_lfam_model(self):
        """Save a model with LFAM and reload it exactly."""
        model = _make_model("fft+srm", use_lfam=True)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _save_checkpoint(model, ckpt_path, branch="fft+srm", use_lfam=True)
            loaded, ck, branch = load_checkpoint(ckpt_path, device="cpu")
            assert branch == "fft+srm"
            assert loaded.use_lfam, "Loaded model should have LFAM enabled"
            x = _rand_face_tensor()
            with torch.no_grad():
                out_orig   = model(x)
                out_loaded = loaded(x)
            assert torch.allclose(out_orig, out_loaded, atol=1e-5), \
                "Loaded model must produce identical outputs"
        finally:
            os.unlink(ckpt_path)

    def test_save_and_load_no_lfam_model(self):
        """Save a model without LFAM (use_lfam=False) and reload it."""
        model = _make_model("fft", use_lfam=False)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _save_checkpoint(model, ckpt_path, branch="fft", use_lfam=False)
            loaded, ck, branch = load_checkpoint(ckpt_path, device="cpu")
            assert not loaded.use_lfam
        finally:
            os.unlink(ckpt_path)

    def test_legacy_checkpoint_loads_without_use_lfam_key(self):
        """Old checkpoints that lack 'use_lfam' default to False (backward compat)."""
        model = _make_model("fft", use_lfam=False)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            # Save WITHOUT use_lfam key (simulating an old checkpoint)
            torch.save({
                "model": model.state_dict(),
                "branch": "fft",
                "use_freq": True,
                "synthetic_ratio": 0.0,
                "threshold": 0.5,
            }, ckpt_path)
            loaded, ck, branch = load_checkpoint(ckpt_path, device="cpu")
            assert not loaded.use_lfam, "Old checkpoints should default to no LFAM"
        finally:
            os.unlink(ckpt_path)

    def test_threshold_preserved_in_checkpoint(self):
        """The decision threshold stored in the checkpoint must survive round-trip."""
        model = _make_model("fft+srm", use_lfam=True)
        CUSTOM_THRESHOLD = 0.62
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _save_checkpoint(model, ckpt_path, threshold=CUSTOM_THRESHOLD)
            _, ck, _ = load_checkpoint(ckpt_path, device="cpu")
            assert abs(ck["threshold"] - CUSTOM_THRESHOLD) < 1e-6
        finally:
            os.unlink(ckpt_path)


# ===========================================================================
# 4. Inference pipeline tests (Detector class)
# ===========================================================================

class TestInference:
    """Tests for the Detector class in inference.py."""

    @pytest.fixture
    def detector_lfam(self, tmp_path):
        """Create a Detector backed by a fresh (random-weights) LFAM model."""
        model = _make_model("fft+srm", use_lfam=True)
        ckpt = tmp_path / "test_lfam.pt"
        _save_checkpoint(model, str(ckpt), branch="fft+srm", use_lfam=True, threshold=0.5)
        return Detector(str(ckpt), device="cpu")

    @pytest.fixture
    def detector_no_lfam(self, tmp_path):
        """Create a Detector backed by a fresh no-LFAM model."""
        model = _make_model("fft", use_lfam=False)
        ckpt = tmp_path / "test_noLFAM.pt"
        _save_checkpoint(model, str(ckpt), branch="fft", use_lfam=False)
        return Detector(str(ckpt), device="cpu")

    def test_predict_returns_three_tuple(self, detector_lfam):
        """predict() must return (prob, overlay, branch_weights)."""
        face = _rand_bgr_face()
        result = detector_lfam.predict(face)
        assert len(result) == 3, "predict() should return (prob, overlay, weights)"
        prob, overlay, weights = result
        assert isinstance(prob, float)
        assert isinstance(overlay, np.ndarray)
        assert isinstance(weights, dict)

    def test_predict_prob_in_range(self, detector_lfam):
        """P(fake) must be in [0, 1]."""
        face = _rand_bgr_face()
        prob, _, _ = detector_lfam.predict(face)
        assert 0.0 <= prob <= 1.0, f"P(fake) out of range: {prob}"

    def test_predict_overlay_same_size_as_input(self, detector_lfam):
        """Grad-CAM overlay must be the same spatial size as the input face."""
        face = _rand_bgr_face(size=224)
        _, overlay, _ = detector_lfam.predict(face)
        assert overlay.shape == face.shape, \
            f"Overlay shape {overlay.shape} != face shape {face.shape}"

    def test_predict_branch_weights_with_lfam(self, detector_lfam):
        """Branch weights dict should have correct keys and values summing to ~1.0."""
        face = _rand_bgr_face()
        _, _, weights = detector_lfam.predict(face)
        assert weights is not None, "LFAM detector should return branch weights"
        assert set(weights.keys()) == {"Spatial", "FFT", "SRM"}, \
            f"Expected keys {{Spatial, FFT, SRM}}, got {set(weights.keys())}"
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4, f"Weights should sum to 1.0, got {total}"

    def test_predict_branch_weights_none_without_lfam(self, detector_no_lfam):
        """Without LFAM, predict() should return None for branch_weights."""
        face = _rand_bgr_face()
        _, _, weights = detector_no_lfam.predict(face)
        assert weights is None, "Non-LFAM detector should return None for branch_weights"

    def test_predict_fast_returns_float(self, detector_lfam):
        """predict_fast() should still return a plain float."""
        face = _rand_bgr_face()
        prob = detector_lfam.predict_fast(face)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_branch_names_match_model_branch(self, detector_lfam):
        """Branch names registered on the detector must match the model branch."""
        assert detector_lfam.branch == "fft+srm"
        assert detector_lfam._branch_names == ["Spatial", "FFT", "SRM"]

    def test_unvalidated_checkpoint_does_not_claim_verified(self, detector_lfam):
        assert detector_lfam.is_validated is False

    def test_no_face_is_not_treated_as_a_center_crop(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        assert detect_box(image) is None

    def test_detector_uses_checkpoint_input_size(self, detector_lfam):
        assert detector_lfam.input_size == 224

    def test_multiple_predictions_consistent(self, detector_lfam):
        """Two calls with the same face should give identical results (eval mode)."""
        face = _rand_bgr_face()
        p1, _, w1 = detector_lfam.predict(face)
        p2, _, w2 = detector_lfam.predict(face)
        assert abs(p1 - p2) < 1e-5, "Predictions should be deterministic"
        for k in w1:
            assert abs(w1[k] - w2[k]) < 1e-5, f"Weight for '{k}' should be deterministic"


# ===========================================================================
# 5. Utility function tests
# ===========================================================================

class TestUtilities:

    def test_to_tensor_shape(self):
        """to_tensor should produce (1, 3, 224, 224) from a 224×224 BGR face."""
        face = _rand_bgr_face(224)
        t = to_tensor(face)
        assert t.shape == (1, 3, 224, 224)

    def test_to_tensor_uses_model_input_size(self):
        face = _rand_bgr_face(224)
        assert to_tensor(face, size=128).shape == (1, 3, 128, 128)

    def test_to_tensor_normalised(self):
        """Tensor values after ImageNet normalisation can exceed [0,1]."""
        face = _rand_bgr_face(224)
        t = to_tensor(face)
        # Just check it's float and doesn't crash
        assert t.dtype == torch.float32

    def test_fft_magnitude_output_shape(self):
        """fft_magnitude should return (B,1,H,W) from (B,3,H,W) input."""
        x = torch.randn(2, 3, 224, 224)
        mag = fft_magnitude(x)
        assert mag.shape == (2, 1, 224, 224)

    def test_topk_mean_full_sequence(self):
        """topk_mean with frac=1.0 should equal the plain mean."""
        probs = [0.1, 0.5, 0.9, 0.4, 0.7]
        assert abs(topk_mean(probs, frac=1.0) - float(np.mean(probs))) < 1e-6

    def test_topk_mean_selects_highest(self):
        """topk_mean with frac=0.2 (1 element from 5) should return the max."""
        probs = [0.1, 0.5, 0.9, 0.4, 0.7]
        result = topk_mean(probs, frac=0.2)
        assert abs(result - 0.9) < 1e-6, f"Expected 0.9, got {result}"

    def test_is_uncertain_true(self):
        """Score within margin of threshold should be uncertain."""
        assert is_uncertain(0.55, 0.5, margin=0.1) is True

    def test_is_uncertain_false(self):
        """Score well above threshold should NOT be uncertain."""
        assert is_uncertain(0.95, 0.5, margin=0.1) is False

    def test_branch_names_dict_covers_all_branches(self):
        """BRANCH_NAMES should have an entry for every branch configuration."""
        for b in BRANCHES:
            assert b in BRANCH_NAMES, f"Missing BRANCH_NAMES entry for '{b}'"


# ===========================================================================
# 6. Model-label / model-tag tests
# ===========================================================================

class TestModelMetadata:

    def test_model_label_lfam_suffix(self):
        label = model_label("fft+srm", use_lfam=True)
        assert "LFAM" in label, f"LFAM should appear in label, got: {label}"

    def test_model_label_no_lfam_no_suffix(self):
        label = model_label("fft", use_lfam=False)
        assert "LFAM" not in label

    def test_model_label_spatial_no_lfam(self):
        """Spatial-only model never has LFAM, even if use_lfam=True."""
        label = model_label("spatial", use_lfam=True)
        assert "LFAM" not in label

    def test_model_tag_lfam_suffix(self):
        tag = model_tag("fft+srm", use_lfam=True)
        assert "lfam" in tag

    def test_model_tag_no_lfam_no_suffix(self):
        tag = model_tag("fft", use_lfam=False)
        assert "lfam" not in tag

    def test_model_tag_synthetic_combo(self):
        tag = model_tag("fft+srm", synthetic_ratio=0.5, use_lfam=True)
        assert "lfam" in tag and "synth" in tag


# ===========================================================================
# Stand-alone runner (no pytest required)
# ===========================================================================

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestLFAM,
        TestFaceSwapNet,
        TestUtilities,
        TestModelMetadata,
    ]

    passed = failed = 0
    print("\n" + "=" * 65)
    print("  LFAM Test Suite (standalone mode)")
    print("=" * 65)

    for cls in test_classes:
        print(f"\n[ {cls.__name__} ]")
        instance = cls()
        for name in dir(cls):
            if not name.startswith("test_"):
                continue
            method = getattr(instance, name)
            try:
                method()
                print(f"  ✔  {name}")
                passed += 1
            except Exception as exc:
                print(f"  ✘  {name}")
                traceback.print_exc()
                failed += 1

    # Checkpoint + Inference tests need a tmp dir – run via pytest for those
    print(f"\n{'='*65}")
    print(f"  Standalone results: {passed} passed, {failed} failed")
    if failed:
        print("  (Run 'python -m pytest test_lfam.py -v' for Detector fixture tests)")
    print("=" * 65)
    sys.exit(failed)
