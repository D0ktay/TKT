import json

import pytest

from tkt.vision import calibration
from tkt.vision.calibration import SurfaceCalibration

# Eğik bakış: uzak kenar görüntüde dar ve yukarıda, yakın kenar geniş ve aşağıda.
TRAPEZOID = ((0.35, 0.20), (0.65, 0.20), (0.85, 0.80), (0.15, 0.80))


def approx(p, q, tol=1e-6):
    return p[0] == pytest.approx(q[0], abs=tol) and p[1] == pytest.approx(q[1], abs=tol)


@pytest.mark.parametrize("image_aspect", [1.0, 640 / 512, 512 / 640, 16 / 9])
def test_corners_map_to_unit_square(image_aspect):
    s = SurfaceCalibration(TRAPEZOID, image_aspect=image_aspect)
    for corner, target in zip(TRAPEZOID, [(0, 0), (1, 0), (1, 1), (0, 1)], strict=True):
        assert approx(s.to_surface(corner), target)


def test_image_aspect_does_not_change_result():
    a = SurfaceCalibration(TRAPEZOID, image_aspect=1.0)
    b = SurfaceCalibration(TRAPEZOID, image_aspect=1.6)
    for p in [(0.5, 0.5), (0.3, 0.7), (0.9, 0.1)]:
        assert approx(a.to_surface(p), b.to_surface(p))


def test_inside_and_outside():
    s = SurfaceCalibration(TRAPEZOID, image_aspect=1.25)
    u, v = s.to_surface((0.5, 0.5))
    assert 0 < u < 1 and 0 < v < 1
    assert SurfaceCalibration.contains((u, v))
    # Görüntü ortasında yatayda, uzak kenarın üstünde: v < 0
    assert s.to_surface((0.5, 0.1))[1] < 0
    # Yakın kenarın altında: v > 1
    assert s.to_surface((0.5, 0.9))[1] > 1
    # Uzak kenar yüksekliğinde ama dar kenarın dışında: u < 0 / u > 1
    assert s.to_surface((0.3, 0.25))[0] < 0
    assert s.to_surface((0.7, 0.25))[0] > 1
    assert not SurfaceCalibration.contains(s.to_surface((0.5, 0.1)))
    assert SurfaceCalibration.contains((1.02, 0.5), margin=0.03)
    assert not SurfaceCalibration.contains(None)


def test_perspective_correction_equal_physical_steps():
    s = SurfaceCalibration(TRAPEZOID, image_aspect=1.25)
    # Uzak ve yakın kenarda aynı fiziksel adım (alan biriminde 0.1): görüntüde farklı uzunluk,
    # alanda aynı fark.
    far = [s.to_image((u, 0.1)) for u in (0.4, 0.5)]
    near = [s.to_image((u, 0.9)) for u in (0.4, 0.5)]
    assert abs(far[1][0] - far[0][0]) < abs(near[1][0] - near[0][0])
    du_far = s.to_surface(far[1])[0] - s.to_surface(far[0])[0]
    du_near = s.to_surface(near[1])[0] - s.to_surface(near[0])[0]
    assert du_far == pytest.approx(0.1, abs=1e-6) and du_near == pytest.approx(0.1, abs=1e-6)


def test_values_beyond_horizon_stay_finite():
    s = SurfaceCalibration(TRAPEZOID)
    for p in [(0.5, -5.0), (0.5, -100.0), (50.0, 0.5), (-50.0, -50.0)]:
        u, v = s.to_surface(p)
        assert calibration.OUTSIDE_MIN <= u <= calibration.OUTSIDE_MAX
        assert calibration.OUTSIDE_MIN <= v <= calibration.OUTSIDE_MAX
        assert not SurfaceCalibration.contains((u, v))


@pytest.mark.parametrize("corners, match", [
    (TRAPEZOID[::-1], "ters"),                                             # ters sıra
    ((TRAPEZOID[0], TRAPEZOID[2], TRAPEZOID[1], TRAPEZOID[3]), "dışbükey"),  # çapraz
    (((0.2, 0.2), (0.8, 0.2), (0.3, 0.3), (0.2, 0.8)), "dışbükey"),       # içbükey
    (((0.5, 0.5), (0.55, 0.5), (0.55, 0.55), (0.5, 0.55)), "küçük"),      # çok küçük
    (((0.2, 0.2), (0.5, 0.2), (0.8, 0.2), (0.2, 0.8)), "dışbükey"),       # üç nokta doğrusal
    (((0.2, 0.2), (0.8, 0.2), (0.8, 0.8)), "4"),                         # eksik köşe
    (((0.2, 0.2), (0.8, 0.2), (0.8, float("nan")), (0.2, 0.8)), "sonlu"),
])
def test_bad_quad_raises(corners, match):
    with pytest.raises(ValueError, match=match):
        SurfaceCalibration(corners)


def test_bad_aspect_raises():
    with pytest.raises(ValueError):
        SurfaceCalibration(TRAPEZOID, aspect=0.0)
    with pytest.raises(ValueError):
        SurfaceCalibration(TRAPEZOID, image_aspect=-1.0)


def test_save_and_load(tmp_path):
    path = tmp_path / "sub" / "surface.json"
    s = SurfaceCalibration(TRAPEZOID, aspect=1.4, image_aspect=1.25)
    calibration.save(s, path)
    data = json.loads(path.read_text())
    assert data["aspect"] == 1.4 and len(data["corners_img"]) == 4
    loaded = calibration.load(path)
    assert loaded == s
    assert approx(loaded.to_surface((0.5, 0.5)), s.to_surface((0.5, 0.5)))


def test_load_missing_returns_none(tmp_path):
    assert calibration.load(tmp_path / "yok.json") is None


def test_load_broken_raises(tmp_path):
    path = tmp_path / "surface.json"
    path.write_text("{bozuk")
    with pytest.raises(ValueError):
        calibration.load(path)
    path.write_text(json.dumps({"aspect": 1.5}))
    with pytest.raises(ValueError):
        calibration.load(path)
    path.write_text(json.dumps({"corners_img": list(map(list, TRAPEZOID[::-1]))}))
    with pytest.raises(ValueError, match="ters"):
        calibration.load(path)
