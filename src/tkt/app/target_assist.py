"""C için pencere içi küçük hedef deneyi (pinch üreten vision kaynağı gerekir)."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import cv2

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.dry_backend import DryBackend
from tkt.vision.debug_view import draw_skeleton, to_bgr
from tkt.vision.live_source import open_live_source

WINDOW = "TKT | Hedef hassasiyeti"


def nearest(point, rect):
    x, y, size = rect
    return (max(x, min(point[0], x + size - 1)), max(y, min(point[1], y + size - 1)))


def candidate(point, targets, radius):
    nearby = []
    for index, rect in enumerate(targets):
        inside = nearest(point, rect)
        if inside == point:
            return index, point
        if math.dist(point, inside) <= radius:
            nearby.append((index, inside))
    return nearby[0] if len(nearby) == 1 else (None, point)


def schedule(radius, repeats, seed):
    jobs = []
    for _ in range(repeats):
        for size in (12, 16, 32):
            for layout in ("alone", "pair", "pass"):
                for direction in (-1, 1):
                    for r in (0, radius):
                        jobs.append((size, layout, direction, r))
    random.Random(seed).shuffle(jobs)
    return jobs


def run(args, log):
    cfg = load_control_config(args.control)
    cfg["general"]["click_method"] = "index_pinch"
    if args.gain_min is not None:
        cfg["pointer"]["gain_min"] = args.gain_min
    jobs = schedule(args.radius, args.repeats, args.seed)
    source = open_live_source(args.vision)
    backend = DryBackend()
    engine = GestureEngine(cfg, backend, source.info)
    rows = []
    ready = False
    try:
        for frame, hand, image in source.frames_raw():
            if hand is not None and frame.pinch is None:
                raise SystemExit("Pinch verisi yok. Kalibre C vision kaynagini kullanin (PR #45).")
            size, layout, direction, radius = jobs[len(rows)]
            target = (480 + direction * 180, 270, size)
            targets = [target]
            if layout != "alone":
                side = -direction if layout == "pass" else direction
                targets.append((target[0] + side * (size + 8), target[1], size))
            if not ready:
                cursor = (480, 270)
                onset = previous = choice = None
                prior_cursor = cursor
                open_since = None
                started = frame.t
                ready = True
            before = cursor
            backend.events.clear()
            backend.set_time(frame.t)
            engine.process(frame)
            events = [ev.args for ev in backend.events if ev.kind == "button"]
            for ev in backend.events:
                if ev.kind == "move":
                    cursor = tuple(
                        max(0, min(limit, p + delta))
                        for p, delta, limit in zip(cursor, ev.args, (959, 539), strict=True)
                    )
            distance = frame.pinch["index"] if frame.present and frame.pinch else None
            if distance is not None and distance <= cfg["index_pinch"]["closed"] and onset is None:
                previous, onset = prior_cursor, before
                choice = candidate(onset, targets, radius)
            click = events == [("left", True), ("left", False)]
            view = cv2.resize(to_bgr(image), (960, 540))
            view = cv2.convertScaleAbs(view, alpha=0.8)
            if hand:
                draw_skeleton(view, hand)
            for idx, (x, y, w) in enumerate(targets):
                color = (85, 225, 110) if idx == 0 else (145, 145, 145)
                cv2.rectangle(view, (x, y), (x + w - 1, y + w - 1), color, -1)
                if choice and choice[0] == idx and radius:
                    cv2.rectangle(view, (x - 2, y - 2), (x + w + 1, y + w + 1), (255, 255, 255), 1)
            cv2.drawMarker(view, cursor, (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            status = f"{len(rows) + 1}/{len(jobs)}   Yesil hedefe tikla   |   Esc: cik"
            cv2.putText(view, status, (16, 34), 0, 0.8, (0, 0, 0), 4)
            cv2.putText(view, status, (16, 34), 0, 0.8, (255, 255, 255), 1)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 255
            if click:
                hit, final = (
                    choice
                    if radius and choice and choice[0] is not None
                    else candidate(cursor, targets, 0)
                )
                row = dict(
                    gain_min=cfg["pointer"]["gain_min"],
                    size=size,
                    layout=layout,
                    radius=radius,
                    targets=targets,
                    before=previous,
                    onset=onset,
                    release=cursor,
                    final=final,
                    hit=hit,
                    seconds=frame.t - started,
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                rows.append(row)
                engine.release()
                ready = False
            else:
                if distance is not None and distance >= cfg["index_pinch"]["open"]:
                    if open_since is None:
                        open_since = frame.t
                else:
                    open_since = None
                released = (
                    open_since is not None
                    and frame.t - open_since >= cfg["index_pinch"]["debounce_ms"] / 1000
                )
                if events or engine.left_down or distance is None or released:
                    onset = previous = choice = None
            prior_cursor = cursor
            if len(rows) == len(jobs) or key in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        source.close()
        cv2.destroyAllWindows()
    for r in (0, args.radius):
        group = [row for row in rows if row["radius"] == r]
        if group:
            print(
                f"{r}px: {sum(row['hit'] == 0 for row in group)}/{len(group)} isabet, "
                f"{sum(row['hit'] == 1 for row in group)} komsu, "
                f"medyan {statistics.median(row['seconds'] for row in group):.2f} sn"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gain-min", type=float, help="Bu oturumun yavas hareket kazanci")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.radius <= 0 or args.repeats < 1:
        parser.error("radius ve repeats pozitif olmali")
    if args.gain_min is not None and (not math.isfinite(args.gain_min) or args.gain_min <= 0):
        parser.error("gain-min sonlu ve pozitif olmali")
    with args.output.open("x", encoding="utf-8") as log:
        run(args, log)


if __name__ == "__main__":
    main()
