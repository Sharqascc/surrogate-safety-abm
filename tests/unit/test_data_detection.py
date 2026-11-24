"""Differential tests for data.detection Detection properties."""
import pytest
from surrogate_safety_abm.data.detection import Detection


def _det(x1, y1, x2, y2):
    return Detection(class_id=1, class_name="sedan", analytical_class="car",
                     confidence=0.9, x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.mark.parametrize("box, want_centroid, want_w, want_h, want_a", [
    ((10, 20, 50, 100), (30.0, 100.0), 40.0, 80.0, 2.0),
    ((50, 20, 50, 100), (50.0, 100.0), 0.0,  80.0, 0.0),
    ((-10, -20, 10, 20), (0.0, 20.0),  20.0, 40.0, 2.0),
    ((0, 0, 1e6, 1e6),   (5e5, 1e6),   1e6,  1e6,  1.0),
])
def test_detection_properties(box, want_centroid, want_w, want_h, want_a):
    d = _det(*box)
    assert d.centroid == pytest.approx(want_centroid)
    assert d.width == pytest.approx(want_w)
    assert d.height == pytest.approx(want_h)
    assert d.aspect == pytest.approx(want_a)
