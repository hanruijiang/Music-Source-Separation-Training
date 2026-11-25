"""CPU-only tests for the shared separation scoring API."""

import numpy as np
import pytest

from utils.metrics import score_tracks


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [1, 2])
def test_matches_original_numpy_sdr(dtype, channels):
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(channels, 257)).astype(dtype)
    estimate = (reference + rng.normal(scale=0.2, size=reference.shape)).astype(dtype)
    # Original NumPy implementation from utils/metrics.py at 1523206.
    refs, ests = reference[None, ...], estimate[None, ...]
    num = np.sum(np.square(refs), axis=(1, 2))
    den = np.sum(np.square(refs - ests), axis=(1, 2))
    num += 1e-8
    den += 1e-8
    expected = float((10 * np.log10(num / den))[0])
    actual = score_tracks(reference, estimate)
    assert actual == {"sdr": expected}
    assert isinstance(actual["sdr"], float)


def test_silence_and_perfect_estimate():
    silence = np.zeros((2, 8), dtype=np.float32)
    reference = np.ones_like(silence)
    assert score_tracks(silence, silence) == {"sdr": 0.0}
    assert score_tracks(reference, silence)["sdr"] == pytest.approx(0.0)
    assert score_tracks(reference, reference)["sdr"] == pytest.approx(
        10 * np.log10((16 + 1e-8) / 1e-8)
    )
    assert score_tracks(silence, reference)["sdr"] < 0


@pytest.mark.parametrize("shapes", [((4,), (4,)), ((1, 4), (2, 4)),
                                     ((2, 4), (2, 3)), ((2, 0), (2, 0)),
                                     ((0, 4), (0, 4)), ((1, 2, 3), (1, 2, 3))])
def test_rejects_invalid_shapes(shapes):
    with pytest.raises(ValueError):
        score_tracks(*(np.zeros(shape) for shape in shapes))


@pytest.mark.parametrize("dtype", [np.int16, np.float16])
def test_low_precision_waveforms(dtype):
    reference = np.full((2, 8), 1000, dtype=dtype)
    estimate = np.full_like(reference, 500)
    assert score_tracks(reference, estimate)["sdr"] == pytest.approx(
        10 * np.log10(4)
    )


def test_rejects_complex_samples():
    with pytest.raises(TypeError):
        score_tracks(np.ones((1, 4), dtype=complex), np.ones((1, 4)))
