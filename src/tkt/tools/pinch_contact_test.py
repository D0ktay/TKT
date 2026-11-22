"""Başparmak-orta parmak temasının algılanma testi (tek pencere, boşlukla başlayan denemeler).

Pozlar: orta parmak teması, işaret parmağı teması (karşılaştırma), açık el. Her denemede
1 sn hazırlık + 1.5 sn kayıt. Her karede `HandFrame.pinch` (0 değiyor, 1 açık) yazılır.
Tuşlar: BOŞLUK denemeyi başlatır, r baştan başlatır, g son denemeyi siler ve geri döner, Esc çıkış.
"""
import json
import sys
import time
from pathlib import Path

import cv2

from tkt.vision.live_source import open_live_source

PREP_S, REC_S = 1.0, 1.5
POSES = {
    "middle": "BASPARMAK -> ORTA PARMAGA DEGSIN, tut",
    "index": "BASPARMAK -> ISARET PARMAGINA DEGSIN, tut",
    "open": "ELIN ACIK, parmaklar ayrik",
}
ORDER = ["middle", "index", "open"] * 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
         (18, 19), (19, 20), (0, 17)]

DEFAULT_OUT = "data/samples/airC/middle-test-%Y%m%d-%H%M%S.jsonl"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(time.strftime(DEFAULT_OUT))
out.parent.mkdir(parents=True, exist_ok=True)
src = open_live_source(str(Path(__file__).resolve().parents[3] / "config/vision-front.toml"))
win = "TKT | orta parmak testi"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)
trial, phase, t0 = 0, "wait", 0.0
records = []
if True:
    for frame, hand, img in src.frames_raw():
        now = time.monotonic()
        vis = cv2.flip(img, 1)                      # ayna: elin sola giderse görüntüde de sola
        h, w = vis.shape[:2]
        if hand is not None:
            pts = [(int((1 - x) * w), int(y * h)) for x, y in hand.image_pts]
            for a, b in EDGES:
                cv2.line(vis, pts[a], pts[b], (80, 220, 120), 3)
            for i, p in enumerate(pts):
                cv2.circle(vis, p, 7 if i in (4, 8, 12) else 4,
                           (0, 220, 255) if i in (4, 8, 12) else (255, 255, 255), -1)
        pose = ORDER[trial] if trial < len(ORDER) else None
        if phase == "prep" and now - t0 >= PREP_S:
            phase, t0 = "rec", now
        if phase == "rec":
            records.append({"trial": trial, "pose": pose, "t": frame.t,
                            "present": frame.present, "pinch": frame.pinch})
            if now - t0 >= REC_S:
                trial, phase = trial + 1, "wait"
        if trial >= len(ORDER):
            break
        pose = ORDER[trial]
        pin = frame.pinch or {}
        cv2.rectangle(vis, (0, 0), (w, 120), (0, 0, 0), -1)
        head = {"wait": f"{trial + 1}/{len(ORDER)}  {POSES[pose]}  -> hazir olunca BOSLUK",
                "prep": f"{trial + 1}/{len(ORDER)}  {POSES[pose]}  -> hazirlan...",
                "rec": f"{trial + 1}/{len(ORDER)}  KAYIT: {POSES[pose]}"}[phase]
        col = {"wait": (255, 255, 255), "prep": (0, 220, 255), "rec": (60, 60, 255)}[phase]
        cv2.putText(vis, head, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
        info = "  ".join(f"{k}={pin[k]:.2f}" for k in ("index", "middle") if k in pin) or "el yok"
        cv2.putText(vis, f"mesafe (0=degiyor 1=acik): {info}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(vis, "BOSLUK: basla   r: bastan   g: son denemeyi tekrarla   Esc: cik",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(" ") and phase == "wait":
            phase, t0 = "prep", now
        elif key == ord("r"):
            records, trial, phase = [], 0, "wait"
            print("bastan basladi", flush=True)
        elif key == ord("g"):
            back = trial if phase != "wait" else max(0, trial - 1)
            records = [x for x in records if x["trial"] < back]
            trial, phase = back, "wait"
            print(f"geri: deneme {back + 1}", flush=True)
src.close()
cv2.destroyAllWindows()
with out.open("w") as fh:
    for rec in records:
        fh.write(json.dumps(rec) + "\n")
print(f"bitti: {trial} deneme -> {out}", flush=True)
