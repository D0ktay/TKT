"""Tap kayıtlarını bip zamanlarına göre değerlendir."""

import argparse
import json
import tomllib
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import ceil, hypot, isfinite
from pathlib import Path
from statistics import median

from tkt.app.config import load_control_config
from tkt.control.gestures import GestureEngine
from tkt.control.os_input.dry_backend import DryBackend
from tkt.control.sources import ReplaySource
from tkt.core.io import prompts_path, read_prompts

# PART1 1.5 ve PART3 3.5 değerlendirme ölçütleri.
_WINDOW_S = 1.5
_CAPTURE_MIN = {"tap1": 48 / 50, "tap2": 45 / 50}
_BUTTON_KIND = {"left": "tap1", "right": "tap2"}
_FALSE_PER_MIN_MAX = 1 / 5
_DRIFT_MAX_PX = 3
_TAP_DURATION_METHOD = (
    "İlk geri alma hareketi (lift eşiğinin aşıldığı karede) kalkış, tık iniş kabul edilir. "
    "Geri alınacak hareket yoksa lift eşiğinin ilk aşıldığı kare kullanılır."
)


@dataclass(frozen=True)
class _Click:
    t: float
    button: str
    drift_px: float | None
    tap_duration_ms: float | None


def build_parser() -> argparse.ArgumentParser:
    """Kayıt, bip zamanları ve yapılandırma yollarını tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.tools.eval_taps")
    parser.add_argument("file", type=Path, nargs="+", help="Değerlendirilecek JSONL kayıtları")
    parser.add_argument("--prompts", type=Path, help="Tek kayıt için özel .prompts.json yolu")
    parser.add_argument(
        "--negative", type=Path, action="append", default=[], metavar="FILE",
        help="Girdi listesindeki bir kaydı tap'siz say ve bip dosyasını yok say (tekrarlanabilir)",
    )
    parser.add_argument("--config", type=Path, default=Path("config/control.toml"))
    parser.add_argument("--click-method", choices=("thumb", "tap", "both"), default="tap")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--grid", nargs="+", metavar="KEY=VALUE,VALUE")
    parser.add_argument("--json", action="store_true")
    return parser


def _override(config, assignment):
    path, separator, raw = assignment.partition("=")
    section, dot, key = path.partition(".")
    if not separator or not dot or key not in config.get(section, {}):
        raise ValueError(f"Bilinmeyen ayar veya geçersiz SECTION.KEY=VALUE: {assignment}")
    old = config[section][key]
    try:
        value = tomllib.loads("value = " + raw)["value"]
    except tomllib.TOMLDecodeError:
        value = raw
    if type(old) in (int, float):
        valid = type(value) in (int, float) and isfinite(value)
    else:
        valid = type(value) is type(old)
    if not valid:
        raise ValueError(f"Geçersiz ayar değeri: {assignment}")
    if path == "general.click_method" and value not in ("thumb", "tap", "both"):
        raise ValueError(f"Geçersiz tık yöntemi: {value}")
    config[section][key] = value


def _replay(path, config):
    source = ReplaySource(path, fast=True)
    backend = DryBackend()
    engine = GestureEngine(config, backend, source.info)
    tap = config["tap"]
    confidence_min = config["general"]["confidence_min"]
    rollback_s = config["pointer"]["rollback_ms"] / 1000
    pair_s = tap["pair_ms"] / 1000
    max_s = tap["max_ms"] / 1000
    frame_times, frame_ids, clicks = [], [], []
    rollback_deadlines, positions = [], []
    dx = dy = 0
    starts = {}
    landed = {}
    try:
        for frame in source.frames():
            t = frame.t
            frame_times.append(t)
            frame_ids.append(frame.frame_id)
            offset = len(backend.events)
            backend.set_time(t)
            engine.process(frame)
            events = backend.events[offset:]

            # The first move on the lift-start frame is the rollback when there
            # is motion to undo. A zero rollback produces no move event.
            lift_t = next((event.t for event in events if event.kind == "move"), t)
            valid_hand = frame.present and frame.confidence >= confidence_min
            for name in ("index", "middle"):
                finger = frame.fingers.get(name) if valid_hand else None
                lift = finger.lift if finger else None
                if lift is None:
                    starts.pop(name, None)
                    landed.pop(name, None)
                    continue
                if lift > tap["lift_start"] and (name not in starts or name in landed):
                    starts[name] = lift_t
                    landed.pop(name, None)
                elif lift < tap["lift_start"] and name in starts:
                    landed.setdefault(name, t)
                # İkili tap'te ilk inen parmağın başlangıcını eşleme süresince tut.
                if name in landed and t > landed[name] + pair_s:
                    starts.pop(name)
                    landed.pop(name)

            for event in events:
                if event.kind == "move":
                    dx += event.args[0]
                    dy += event.args[1]
                    rollback_deadlines.append(event.t + rollback_s)
                    positions.append((dx, dy))
                elif event.kind == "button":
                    button, down = event.args
                    if not down or button not in _BUTTON_KIND:
                        continue
                    names = ("index",) if button == "left" else ("index", "middle")
                    drift = tap_duration = None
                    if all(name in landed for name in names) and max(landed[n] for n in names) == t:
                        start_t = min(starts[name] for name in names)
                        if t <= start_t + max_s + (pair_s if button == "right" else 0):
                            # The engine rolls back moves exactly at the cutoff too.
                            index = bisect_left(rollback_deadlines, start_t)
                            start_x, start_y = positions[index - 1] if index else (0, 0)
                            drift = hypot(dx - start_x, dy - start_y)
                            tap_duration = (event.t - start_t) * 1000
                    clicks.append(_Click(event.t, button, drift, tap_duration))
    finally:
        source.close()
    return frame_times, frame_ids, clicks


def _false_clicks(duration, clicks):
    return {
        "duration_s": duration,
        "outside_windows_s": duration,
        **clicks,
        "per_minute": sum(clicks.values()) * 60 / duration if duration > 0 else None,
    }


def _measure(path, sidecar, config, negative):
    prompts = []
    if not negative:
        try:
            prompts = sorted(read_prompts(sidecar or prompts_path(path)), key=lambda p: p.t)
        except FileNotFoundError:
            if sidecar is not None:
                raise
            negative = True
        if any(not isfinite(prompt.t) for prompt in prompts):
            raise ValueError(f"Sonlu olmayan bip zamanı: {sidecar or prompts_path(path)}")

    times, ids, clicks = _replay(path, config)
    by_kind = {
        kind: [click for click in clicks if click.button == button]
        for button, kind in _BUTTON_KIND.items()
    }
    click_times = {kind: [click.t for click in events] for kind, events in by_kind.items()}
    next_click = dict.fromkeys(_CAPTURE_MIN, 0)
    remaining = {kind: len(events) for kind, events in by_kind.items()}
    captures, misses = [], []
    for prompt in prompts:
        end = prompt.t + _WINDOW_S
        kind = prompt.kind
        index = bisect_left(click_times[kind], prompt.t, lo=next_click[kind])
        if index < len(by_kind[kind]) and by_kind[kind][index].t <= end:
            click = by_kind[kind][index]
            captures.append({
                "kind": kind,
                "prompt_t": prompt.t,
                "click_t": click.t,
                "latency_ms": (click.t - prompt.t) * 1000,
                "drift_px": click.drift_px,
                "tap_duration_ms": click.tap_duration_ms,
            })
            next_click[kind] = index + 1
            remaining[kind] -= 1
        else:
            first = bisect_left(times, prompt.t)
            frame_id = ids[first] if first < len(times) and times[first] <= end else None
            misses.append({"kind": kind, "t": prompt.t, "frame_id": frame_id})

    unmatched = {button: remaining[kind] for button, kind in _BUTTON_KIND.items()}
    duration = times[-1] - times[0] if times else 0.0
    return {
        "file": str(path),
        "captures": captures,
        "misses": misses,
        "false_clicks": _false_clicks(duration, unmatched) if negative else None,
        "unmatched_clicks": unmatched if not negative else None,
    }


def _time_summary(values):
    values = sorted(values)
    return {
        "median": median(values) if values else None,
        "p95": values[ceil(0.95 * len(values)) - 1] if values else None,
    }


def _report(paths, config, sidecar, negatives):
    recordings = [_measure(path, sidecar, config, path in negatives) for path in paths]
    captures = [item for recording in recordings for item in recording["captures"]]
    misses = [item for recording in recordings for item in recording["misses"]]
    rates = {}
    for kind in _CAPTURE_MIN:
        caught = sum(item["kind"] == kind for item in captures)
        total = caught + sum(item["kind"] == kind for item in misses)
        rates[kind] = {"caught": caught, "total": total, "rate": caught / total if total else None}
    false = [r["false_clicks"] for r in recordings if r["false_clicks"] is not None]
    unmatched = [r["unmatched_clicks"] for r in recordings if r["unmatched_clicks"] is not None]
    false_summary = _false_clicks(
        sum(item["duration_s"] for item in false),
        {button: sum(item[button] for item in false) for button in _BUTTON_KIND},
    )
    drifts = [item["drift_px"] for item in captures if item["drift_px"] is not None]
    durations = [
        item["tap_duration_ms"] for item in captures if item["tap_duration_ms"] is not None
    ]
    checks = {
        kind: data["rate"] is not None and data["rate"] >= _CAPTURE_MIN[kind]
        for kind, data in rates.items()
    }
    per_minute = false_summary["per_minute"]
    checks["false_clicks"] = per_minute is not None and per_minute <= _FALSE_PER_MIN_MAX
    checks["drift"] = bool(drifts) and len(drifts) == len(captures) and max(drifts) < _DRIFT_MAX_PX
    return {
        "capture": rates,
        "false_clicks": false_summary,
        "unmatched_clicks": {
            button: sum(item[button] for item in unmatched) for button in _BUTTON_KIND
        },
        "drift_px": {
            "count": len(drifts),
            "median": median(drifts) if drifts else None,
            "max": max(drifts, default=None),
        },
        "latency_ms": _time_summary(item["latency_ms"] for item in captures),
        "tap_duration_ms": {"count": len(durations), **_time_summary(durations)},
        "tap_duration_method": _TAP_DURATION_METHOD,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "recordings": recordings,
    }


def _number(value, factor=1):
    return "—" if value is None else f"{value * factor:.2f}"


def _click_text(data):
    return "—" if data is None else f"sol {data['left']}, sağ {data['right']}"


def _false_text(data):
    if data is None:
        return "Tap'siz yanlış tık —"
    rate = _number(data["per_minute"])
    return f"Tap'siz yanlış tık {_click_text(data)}, {rate}/dk ({data['duration_s']:.2f} sn)"


def _print_report(report, detailed=True):
    rates = ", ".join(
        f"{kind} {data['caught']}/{data['total']} (%{_number(data['rate'], 100)})"
        for kind, data in report["capture"].items()
    )
    drift, latency, duration = report["drift_px"], report["latency_ms"], report["tap_duration_ms"]
    drift_text = f"Kayma medyan/maks {_number(drift['median'])}/{_number(drift['max'])} px"
    delay = f"Bip→tık medyan/p95 {_number(latency['median'])}/{_number(latency['p95'])} ms"
    tap_time = f"Tap süresi medyan/p95 {_number(duration['median'])}/{_number(duration['p95'])} ms"
    unmatched = f"Bipli eşleşmeyen tık {_click_text(report['unmatched_clicks'])}"
    gate = report["gate"]
    result = "GEÇTİ" if gate["passed"] else "KALDI"
    failed = ", ".join(name for name, passed in gate["checks"].items() if not passed)
    gate_text = f"Kapı {result}" + (f" ({failed})" if failed else "")
    false_text = _false_text(report["false_clicks"])
    print(" | ".join((rates, false_text, unmatched, drift_text, delay, tap_time, gate_text)))
    if detailed:
        print(report["tap_duration_method"])
        for recording in report["recordings"]:
            print(
                f"{recording['file']} | {_false_text(recording['false_clicks'])} | "
                f"Bipli eşleşmeyen tık {_click_text(recording['unmatched_clicks'])}"
            )
            for miss in recording["misses"]:
                print(f"  Kaçan {miss['kind']} t={miss['t']:.6f} frame_id={miss['frame_id']}")
            for capture in recording["captures"]:
                print(
                    f"  Yakalanan {capture['kind']} t={capture['prompt_t']:.6f} "
                    f"kayma={_number(capture['drift_px'])} px "
                    f"bip→tık={_number(capture['latency_ms'])} ms "
                    f"tap süresi={_number(capture['tap_duration_ms'])} ms"
                )


def _grid_axes(specs):
    for spec in specs:
        key, separator, values = spec.partition("=")
        if not separator or not key or not all(values.split(",")):
            raise ValueError(f"Geçersiz ızgara ekseni: {spec}")
        path = key if "." in key else f"tap.{key}"
        yield [f"{path}={value}" for value in values.split(",")]


def main(argv: list[str] | None = None) -> None:
    """Kayıtları oynat ve tap ölçümlerini yazdır."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompts and len(args.file) != 1:
        parser.error("--prompts yalnızca tek kayıtla kullanılabilir")
    negatives = set(args.negative)
    if not negatives.issubset(args.file):
        parser.error("--negative ile işaretlenen kayıtlar girdi listesinde olmalı")
    try:
        config = load_control_config(args.config)
        config["general"]["click_method"] = args.click_method
        for assignment in args.set:
            _override(config, assignment)
        if args.grid:
            results = []
            for assignments in product(*_grid_axes(args.grid)):
                candidate = deepcopy(config)
                for assignment in assignments:
                    _override(candidate, assignment)
                results.append({
                    "settings": assignments,
                    **_report(args.file, candidate, args.prompts, negatives),
                })
        else:
            results = _report(args.file, config, args.prompts, negatives)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.grid:
        print(_TAP_DURATION_METHOD)
        for result in results:
            print(" ".join(result["settings"]), end=" | ")
            _print_report(result, detailed=False)
    else:
        _print_report(results)


if __name__ == "__main__":
    main()
