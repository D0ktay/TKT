# TKT

TKT turns your hand into a pointing device using a single camera. It has two
setups, and an auto mode that switches between them:

| Version | Camera | How you control the cursor |
|---|---|---|
| **A — desk** | Looks down at the desk / mousepad | Hand rests on the desk like on a touchpad: index finger moves the cursor, thumb clicks and drags, two fingers scroll, three or four fingers switch desktops, pinky alone starts dictation. |
| **C — pinch** | Faces you (above or below the screen) | Hand in the air: palm moves the cursor, thumb–index pinch clicks and drags, thumb–middle pinch scrolls. |
| **Auto (A/C)** | Either | Picks the version from what the camera sees: a face in view means C, no face means A. |

Hand tracking runs locally with MediaPipe. macOS and Windows are both targets;
the auto mode, its HUD and system dictation are macOS only for now. Finger
tapping on the desk (J7–J8) and the active desk area are experimental and off by
default.

## Setup

Run all commands from the repository root with Python 3.11 and `uv`. On Intel
Macs MediaPipe is pinned to 0.10.21 in `uv.lock`.

```bash
uv sync
uv run python -m tkt.vision.tracker --download
```

The auto mode downloads the MediaPipe face detector
(`models/blaze_face_short_range.tflite`) on first start.

## Camera and permissions

Connect the LogiInsight kit camera (LVNext) with a USB 3 cable directly to a USB 3
port.

- **Version A:** mount the camera so it looks down at the desk, facing your
  fingertips. Camera settings: [vision.toml](config/vision.toml).
- **Version C:** point the camera at yourself. Camera settings:
  [vision-front.toml](config/vision-front.toml) (1920×1080; the auto mode mirrors horizontal
  movement so the cursor follows your hand).

On macOS the color stream runs at about 30 fps through OpenCV; the IR (CPD) stream
is 512×640, portrait and grayscale. On Windows the app asks DirectShow for MJPG
and falls back to Media Foundation.

```bash
uv run python -m tkt.vision.camera --list
uv run python -m tkt.vision.camera --probe
```

Use `--list` to find the camera, then set `index` (or `name`), `width` and
`height` under `[camera]`. `--probe` measures the real frame rate.

- **macOS:** in System Settings → Privacy & Security, give the terminal or app
  that runs TKT access to Camera, Accessibility and Input Monitoring. Dictation
  also needs Automation permission for System Events (macOS asks on first use)
  and Dictation turned on under Keyboard → Dictation.
- **Windows:** windows running as administrator ignore input sent by a
  non-elevated TKT.

## Running

### Auto mode (A and C, macOS)

```bash
uv run python -m tkt.app.auto_mode
```

- Once a second the camera image is checked for a face. A face with confidence
  ≥ 0.7 in three checks in a row switches to C; three checks without one switch
  back to A. Each version uses its own settings: A uses `vision.toml` and the
  `[pointer]` values with thumb clicks; C uses `vision-front.toml`, `[pointer]`
  overridden by `[pointer_front]`, index pinch and target assist.
- On a switch, held buttons are released, dictation is stopped and the camera is
  reopened with the new settings, so expect a 1–2 s gap.
- The current version is shown in the top-left corner of the screen and the
  gesture state (SCROLL, PINCH, STOP, NO HAND…) next to the cursor.
- **HUD window ("TKT HUD")**: camera image, hand skeleton, current state, which
  fingers are open, and a list of the last 12 actions (clicks, drags, scroll
  amounts, system actions, version switches). In A it also draws the active desk
  area from `~/.tkt/surface.json`: green when the index fingertip is inside,
  red when outside. Closing the window only hides it.
- In C, target assist highlights a single nearby button with a green frame; a
  pinch that starts and ends on the same button clicks its center.
- **Esc** quits and releases all buttons.

### Version A only (macOS and Windows)

```bash
uv run python -m tkt.app.main --source live --backend pynput --hud
uv run python -m tkt.app.main --source live --backend native --latency
```

- `pynput` works on both systems.
- `native` uses `SendInput` on Windows and Quartz on macOS. On macOS this sends
  proper drag events and double-click counts (needs Accessibility permission).
- `--hud` opens a window with the camera image, skeleton and the recognized gesture.
- `--latency` measures the time from frame capture to input and the control
  processing time; the target is p95 < 60 ms.
- `Ctrl+Alt+P` pauses output and releases held buttons; press again to resume.
  Change it with `[general].panic_hotkey`.

### Debug window and calibration

```bash
uv run python -m tkt.vision.debug_view                                  # A
uv run python -m tkt.vision.debug_view --config config/vision-front.toml  # C
```

Shows the skeleton, per-finger curl, lift and pinch bars, the lift graph and fps.

- `c`: hand calibration (open hand, fist, thumb closed; space starts each step).
- `p`: hand calibration plus pinch steps (thumb touching each finger), for C.
- `t`: tap calibration; tap with the index finger each time "TAP!" appears (10 times).
- `m`: desk (mousepad) model: move the hand over the whole pad.
- `s` (experimental): active desk area. Put the index fingertip on the four
  corners in order (far-left, far-right, near-right, near-left) and press space
  at each; `g` redoes a corner. Saved to `[surface].file`.
- `q`, Esc or closing the window: quit.

Calibration takes effect immediately and is written to `[hand].calibration`.

## Gestures

### Version A — desk

Rest your wrist and the base of your palm on the desk. To change gestures, curl
all four fingers first.

| Code | Movement | Result |
|---|---|---|
| J0 | Loose fist | Rest; no gesture starts. |
| J1 | Only the index finger open, slide | Relative, accelerated cursor movement. |
| J2 | Curl the index, move the hand, open again | Reposition without moving the cursor. |
| J3 | Index open, close and open the thumb | Left click. |
| J4 | Index and middle open, hand still, close the thumb | Right click. |
| J5 | Index open, keep the thumb closed and slide | Left drag. |
| J6 | Slide index and middle together | Scrolling (natural direction, with inertia). |
| J7 | Lift the index finger and tap the desk | Experimental. Left click. |
| J8 | Tap with index and middle together | Experimental. Right click. |
| J9 | Three or four fingers open, swipe sideways or up | Switch desktop / Mission Control (Task View on Windows). |
| J10 | Take the hand out of view, or press Ctrl+Alt+P | Output stops; held buttons are released. |
| J11 | Only the pinky open (thumb free) | Starts system dictation; speak. A fist stops dictation and presses Enter. |

On macOS, J9 needs the Mission Control and desktop-switching keyboard shortcuts
enabled. Dictation types into whatever text field has focus, not where the cursor
is; macOS also ends dictation by itself after about 7 seconds of silence.

### Version C — pinch

The camera tracks your palm, so the cursor follows the whole hand.

| Movement | Result |
|---|---|
| Open hand, move | Cursor movement. |
| Thumb touches index briefly | Left click (on the highlighted button if target assist found one). |
| Thumb on index, move the hand | Left drag. |
| Thumb on index, hold still for 2 s | Right click. |
| Thumb on middle, move the hand | Scrolling. |
| Fist | Cursor stops. |

## Recording and evaluation

```bash
uv run python -m tkt.tools.record --out tap.jsonl --raw --prompt 10 --kind tap1
uv run python -m tkt.tools.record --out normal.jsonl --raw --duration 20
uv run python -m tkt.tools.record --out c.jsonl --raw --config config/vision-front.toml --wait-space
uv run python -m tkt.app.main --source replay --file tap.jsonl --backend dry
uv run python -m tkt.tools.replay tap.jsonl
uv run python -m tkt.tools.eval_taps tap.jsonl normal.jsonl
uv run python -m tkt.vision.lift --eval tap.raw.jsonl normal.raw.jsonl
```

- `--raw` also writes the 21 hand points to `*.raw.jsonl`.
- `--prompt N` beeps N times and writes the beep times to `*.prompts.json`; tap
  once per beep. A prompted recording ends shortly after the last beep.
- `--wait-space` starts recording when you press space, so several recordings
  can be made without reopening the camera.
- Without a duration, stop with `q` then Enter, or Ctrl+C.
- `dry` replay sends no input to the OS. `tkt.tools.replay` shows a recording
  frame by frame.
- `eval_taps` counts correct taps per beep and false taps in the normal
  recording. `lift --eval` measures how well taps separate from normal movement.
- `python -m tkt.tools.pinch_contact_test` checks whether thumb–middle and
  thumb–index contact are told apart (C).

Sample recordings are in [data/samples/](data/samples/).

## Settings

| File | Main keys |
|---|---|
| [vision.toml](config/vision.toml) | Version A camera. `[camera]` `index`, `name`, `width`, `height`, `fps`, `backend`, `rotate_180`. `[tracker]` `model`, `min_detection`, `min_tracking`. `[hand]` `side`, `calibration`, `lift_off` (commented out = hand-lift detection off). |
| [vision-front.toml](config/vision-front.toml) | Version C camera; same sections as `vision.toml`. |
| [vision.toml](config/vision.toml) | Experimental: `[surface]` `enabled`, `file`, `aspect` (active desk area). |
| [control.toml](config/control.toml) | `[general]` `panic_hotkey`, `click_method` (`thumb`, `tap`, `both`, `pinch`, `index_pinch`), `confidence_min`, `lost_release_ms`. |
| [control.toml](config/control.toml) | `[mode]` `lock_ms`, `curl_open`, `curl_closed`. `[thumb]` `closed`, `open`, `drag_threshold`. |
| [control.toml](config/control.toml) | `[pointer]` `gain_min`, `gain_max`, `min_cutoff`. `[pointer_front]` overrides `[pointer]` in C. |
| [control.toml](config/control.toml) | `[scroll]` `natural`, `gain`, `step_px`, `inertia_ms`. `[tap]` `lift_start`, `lift_up`, `min_ms`, `max_ms`, `pair_ms`. `[swipe]` `distance`, `window_ms`. |
| [control.toml](config/control.toml) | Version C: `[pinch]` and `[index_pinch]` `closed`, `open`, `debounce_ms`, `drag_threshold`, `hold_ms` (right click), `fist_curl`. |
| [control.toml](config/control.toml) | Experimental: `[surface_pointer]` `enabled`, `aspect`, `gain_*`, `speed_*`, `absolute`. `aspect` must match `[surface]`. |

## Troubleshooting

- **Low fps:** check the USB 3 port and cable, try a lower resolution and
  measure again with `--probe`.
- **Camera does not open:** check camera permissions and whether another app is
  using the camera.
- **Wrong camera opens:** set `[camera].name` or `[camera].index` from the
  `--list` output.
- **Auto mode switches by itself:** a face partly in view (or someone walking
  behind you) can trigger C. Watch the face confidence in the log.
- **Dictation does not start:** turn on Dictation in System Settings, allow
  Automation for System Events, and make sure a text field has focus. The app
  in front must have an Edit menu with "Start Dictation".
- **Distorted movement on a scaled Windows display:** check the display scale and
  start through `tkt.app.main`, which enables DPI awareness.
- **Tap does nothing:** set `[general].click_method` to `tap` or `both` and check
  that the `[hand].calibration` file has valid `tap_ref` values. Hand
  calibration (`c`) does not produce a tap reference.

## Development

```bash
uv run ruff check .
uv run pytest
```

Camera tests are skipped by default. Plans are in [docs/plan/](docs/plan/) and
agent rules in [AGENTS.md](AGENTS.md).
