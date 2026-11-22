"""macOS için çift tıklanan TKT.app oluşturur: python -m tkt.tools.make_mac_app

Uygulama küçük bir kabuktur: bu repo kopyasındaki `.venv` ile `python -m tkt.app.auto_mode`
çalıştırır, çıktıyı `~/Library/Logs/TKT.log` dosyasına yazar. Repo taşınırsa ya da başka bir
kopyadan çalıştırılacaksa yeniden oluşturulur.

İlk açılışta macOS "TKT" için Kamera izni sorar; Erişilebilirlik izni Sistem Ayarları >
Gizlilik ve Güvenlik > Erişilebilirlik'ten verilir (imleç, kısayollar, hedef yardımı).
Dikte için Otomasyon (System Events) izni de istenir.
"""
import argparse
import plistlib
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP_NAME = "TKT"
BUNDLE_ID = "local.tkt.app"

LAUNCHER = """#!/bin/sh
# tkt.tools.make_mac_app üretti; elle düzenleme, yeniden oluştur.
cd "{repo}" || exit 1
mkdir -p "$HOME/Library/Logs"
exec "{python}" -u -m tkt.app.auto_mode >> "$HOME/Library/Logs/TKT.log" 2>&1
"""


def info_plist():
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1",
        "CFBundleVersion": "1",
        "LSUIElement": True,        # Dock simgesi yok; menü çubuğunda "TKT"
        "NSHighResolutionCapable": True,
        "NSCameraUsageDescription": "TKT elini ve yüzünü kamerayla izler.",
        "NSAppleEventsUsageDescription": "TKT dikteyi uygulama menüsünden açıp kapatır.",
    }


def build(dest, repo=REPO, python=None):
    """`dest` klasörüne TKT.app yazar (varsa üstüne); uygulamanın yolunu döner."""
    repo = Path(repo)
    python = Path(python) if python is not None else repo / ".venv/bin/python"
    if not python.exists():
        raise FileNotFoundError(f"python bulunamadı: {python} (önce `uv sync`)")
    app = Path(dest).expanduser() / f"{APP_NAME}.app"
    macos = app / "Contents/MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    with (app / "Contents/Info.plist").open("wb") as fh:
        plistlib.dump(info_plist(), fh)
    launcher = macos / APP_NAME
    launcher.write_text(LAUNCHER.format(repo=repo, python=python))
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tkt.tools.make_mac_app")
    parser.add_argument("--dest", default="~/Applications", help="TKT.app'in konacağı klasör")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("yalnız macOS")
    print(build(args.dest))


if __name__ == "__main__":
    main()
