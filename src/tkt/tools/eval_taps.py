"""Tap kayıtlarını bip zamanlarına göre değerlendir."""

import argparse
import json
import tomllib
from bisect import bisect_left, bisect_right
from copy import deepcopy
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


def build_parser() -> argparse.ArgumentParser:
    """Kayıt, bip zamanları ve yapılandırma yollarını tanımla."""
    parser = argparse.ArgumentParser(prog="python -m tkt.tools.eval_taps")
    parser.add_argument("file", type=Path, nargs="+", help="Değerlendirilecek JSONL kayıtları")
    parser.add_argument("--prompts", type=Path, help="Tek kayıt için özel .prompts.json yolu")
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
    pair_s = tap["pair_ms"] / 1000
    max_s = tap["max_ms"] / 1000
    frame_times, frame_ids, clicks = [], [], []
    dx = dy = 0
    starts = {}
    landed = {}
    try:
        for frame in source.frames():
            t = frame.t
            frame_times.append(t)
            frame_ids.append(frame.frame_id)
            valid_hand = frame.present and frame.confidence >= confidence_min
            for name in ("index", "middle"):
                finger = frame.fingers.get(name) if valid_hand else None
                lift = finger.lift if finger else None
                if lift is None:
                    starts.pop(name, None)
                    landed.pop(name, None)
                    continue
                if lift > tap["lift_start"] and (name not in starts or name in landed):
                    starts[name] = (t, dx, dy)
                    landed.pop(name, None)
                elif lift < tap["lift_start"] and name in starts:
                    landed.setdefault(name, t)
                # İkili tap'te ilk inen parmağın başlangıcını eşleme süresince tut.
                if name in landed and t > landed[name] + pair_s:
                    starts.pop(name)
                    landed.pop(name)

            offset = len(backend.events)
            backend.set_time(t)
            engine.process(frame)
            for event in backend.events[offset:]:
                if event.kind == "move":
                    dx += event.args[0]
                    dy += event.args[1]
                elif event.kind == "button":
                    button, down = event.args
                    if not down or button not in _BUTTON_KIND:
                        continue
                    names = ("index",) if button == "left" else ("index", "middle")
                    drift = None
                    if all(name in starts for name in names):
                        start_t, start_x, start_y = min(starts[name] for name in names)
                        if t <= start_t + max_s + (pair_s if button == "right" else 0):
                            drift = hypot(dx - start_x, dy - start_y)
                    clicks.append((event.t, button, drift))
    finally:
        source.close()
    return frame_times, frame_ids, clicks


def _windows(prompts):
    """Çakışan bip pencerelerini tek aralık olarak say."""
    windows = []
    for prompt in prompts:
        start, end = prompt.t, prompt.t + _WINDOW_S
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    return windows


def _false_clicks(duration, exposure, clicks):
    return {
        "duration_s": duration,
        "outside_windows_s": exposure,
        **clicks,
        "per_minute": sum(clicks.values()) * 60 / exposure if exposure > 0 else None,
    }


def _measure(path, sidecar, config):
    try:
        prompts = sorted(read_prompts(sidecar), key=lambda prompt: prompt.t)
    except FileNotFoundError:
        if sidecar != prompts_path(path):
            raise
        prompts = []
    if any(not isfinite(prompt.t) for prompt in prompts):
        raise ValueError(f"Sonlu olmayan bip zamanı: {sidecar}")

    times, ids, clicks = _replay(path, config)
    by_kind = {
        kind: [(t, drift) for t, name, drift in clicks if name == button]
        for button, kind in _BUTTON_KIND.items()
    }
    click_times = {kind: [t for t, _ in events] for kind, events in by_kind.items()}
    next_click = dict.fromkeys(_CAPTURE_MIN, 0)
    captures, misses = [], []
    for prompt in prompts:
        end = prompt.t + _WINDOW_S
        kind = prompt.kind
        index = bisect_left(click_times[kind], prompt.t, lo=next_click[kind])
        if index < len(by_kind[kind]) and by_kind[kind][index][0] <= end:
            t, drift = by_kind[kind][index]
            captures.append({
                "kind": kind,
                "prompt_t": prompt.t,
                "click_t": t,
                "latency_ms": (t - prompt.t) * 1000,
                "drift_px": drift,
            })
            next_click[kind] = index + 1
        else:
            first = bisect_left(times, prompt.t)
            frame_id = ids[first] if first < len(times) and times[first] <= end else None
            misses.append({"kind": kind, "t": prompt.t, "frame_id": frame_id})

    windows = _windows(prompts)
    starts = [start for start, _ in windows]
    false = dict.fromkeys(_BUTTON_KIND, 0)
    for t, button, _ in clicks:
        index = bisect_right(starts, t) - 1
        if index < 0 or t > windows[index][1]:
            false[button] += 1
    duration = times[-1] - times[0] if times else 0.0
    covered = 0.0
    if times:
        covered = sum(
            max(0.0, min(end, times[-1]) - max(start, times[0]))
            for start, end in windows
        )
    return {
        "file": str(path),
        "captures": captures,
        "misses": misses,
        "false_clicks": _false_clicks(duration, max(0.0, duration - covered), false),
    }


def _report(paths, config, sidecar):
    recordings = [_measure(path, sidecar or prompts_path(path), config) for path in paths]
    captures = [item for recording in recordings for item in recording["captures"]]
    misses = [item for recording in recordings for item in recording["misses"]]
    rates = {}
    for kind in _CAPTURE_MIN:
        caught = sum(item["kind"] == kind for item in captures)
        total = caught + sum(item["kind"] == kind for item in misses)
        rates[kind] = {"caught": caught, "total": total, "rate": caught / total if total else None}
    false = [recording["false_clicks"] for recording in recordings]
    false_summary = _false_clicks(
        sum(item["duration_s"] for item in false),
        sum(item["outside_windows_s"] for item in false),
        {button: sum(item[button] for item in false) for button in _BUTTON_KIND},
    )
    drifts = [item["drift_px"] for item in captures if item["drift_px"] is not None]
    latencies = sorted(item["latency_ms"] for item in captures)
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
        "drift_px": {
            "count": len(drifts),
            "median": median(drifts) if drifts else None,
            "max": max(drifts, default=None),
        },
        "latency_ms": {
            "median": median(latencies) if latencies else None,
            "p95": latencies[ceil(0.95 * len(latencies)) - 1] if latencies else None,
        },
        "gate": {"passed": all(checks.values()), "checks": checks},
        "recordings": recordings,
    }


def _number(value, factor=1):
    return "—" if value is None else f"{value * factor:.2f}"


def _false_text(data):
    rate = _number(data["per_minute"])
    return (
        f"Yanlış tık: sol {data['left']}, sağ {data['right']}, {rate}/dk "
        f"({data['outside_windows_s']:.2f} sn pencere dışı / {data['duration_s']:.2f} sn kayıt)"
    )


def _print_report(report, detailed=True):
    rates = ", ".join(
        f"{kind} {data['caught']}/{data['total']} (%{_number(data['rate'], 100)})"
        for kind, data in report["capture"].items()
    )
    drift, latency = report["drift_px"], report["latency_ms"]
    drift_text = f"Kayma medyan/maks {_number(drift['median'])}/{_number(drift['max'])} px"
    delay = f"Gecikme medyan/p95 {_number(latency['median'])}/{_number(latency['p95'])} ms"
    gate = report["gate"]
    result = "GEÇTİ" if gate["passed"] else "KALDI"
    failed = ", ".join(name for name, passed in gate["checks"].items() if not passed)
    gate_text = f"Kapı {result}" + (f" ({failed})" if failed else "")
    print(" | ".join((rates, _false_text(report["false_clicks"]), drift_text, delay, gate_text)))
    if detailed:
        for recording in report["recordings"]:
            print(f"{recording['file']} | {_false_text(recording['false_clicks'])}")
            for miss in recording["misses"]:
                print(f"  Kaçan {miss['kind']} t={miss['t']:.6f} frame_id={miss['frame_id']}")
            for capture in recording["captures"]:
                print(
                    f"  Yakalanan {capture['kind']} t={capture['prompt_t']:.6f} "
                    f"kayma={_number(capture['drift_px'])} px "
                    f"gecikme={_number(capture['latency_ms'])} ms"
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
                    **_report(args.file, candidate, args.prompts),
                })
        else:
            results = _report(args.file, config, args.prompts)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.grid:
        for result in results:
            print(" ".join(result["settings"]), end=" | ")
            _print_report(result, detailed=False)
    else:
        _print_report(results)


if __name__ == "__main__":
    main()
