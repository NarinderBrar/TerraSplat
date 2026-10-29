import numpy as np
import pytest

from pipeline.evaluation import image_metrics


def test_masked_metrics_exclude_padding_and_count_pixels_not_channels():
    reference = np.zeros((2, 2, 3))
    rendered = np.full((2, 2, 3), 100.0)
    rendered[0, 0] = [1, 0, 0]
    mask = np.array([[1, 0], [0, 0]])
    result = image_metrics(reference, rendered, mask)
    assert result["valid_pixels"] == 1
    assert result["mse"] == pytest.approx(1 / 3)
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["psnr_db"] == pytest.approx(10 * np.log10(3))


def test_nonfinite_render_is_rejected_instead_of_hidden_by_mask():
    image = np.zeros((2, 2, 3))
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        image_metrics(np.zeros_like(image), image, np.array([[0, 1], [1, 1]]))
