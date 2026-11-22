"""Control testleri için sentetik el kareleri."""

from collections.abc import Mapping

from tkt.core.types import FINGERS, Finger, HandFrame


def make_frame(
    *,
    t: float = 0.0,
    frame_id: int = 0,
    present: bool = True,
    curls: Mapping[str, float] | None = None,
    lift: float | Mapping[str, float | None] | None = None,
    thumb_out: float = 1.0,
    palm: tuple[float, float] = (0.5, 0.6),
    index_pos: tuple[float, float] | None = None,
    positions: Mapping[str, tuple[float, float]] | None = None,
    surfaces: Mapping[str, tuple[float, float] | None] | None = None,
    hand_size: float = 0.2,
    confidence: float = 1.0,
    pinch: Mapping[str, float] | None = None,
) -> HandFrame:
    """Eksik parmak ölçümlerini varsayılanlarla doldur ve geçerli bir kare kur.

    `pinch` verilmezse kare pinch taşımaz (None). Verilirse eksik parmaklar 1.0 (açık)
    ile doldurulur; `present=False` ise yok sayılır.
    """
    curls = curls or {}
    positions = positions or {}
    surfaces = surfaces or {}
    fingers = {}
    if present:
        for name in FINGERS:
            img = positions.get(name, (0.5, 0.4))
            if name == "index" and index_pos is not None:
                img = index_pos
            fingers[name] = Finger(
                img=img,
                surface=surfaces.get(name),
                curl=curls.get(name, 0.0),
                lift=lift.get(name) if isinstance(lift, Mapping) else lift,
            )

    return HandFrame(
        t=t,
        frame_id=frame_id,
        present=present,
        fingers=fingers,
        thumb_out=thumb_out,
        palm=palm,
        hand_size=hand_size,
        confidence=confidence,
        pinch=(
            {name: pinch.get(name, 1.0) for name in FINGERS}
            if present and pinch is not None
            else None
        ),
    )
