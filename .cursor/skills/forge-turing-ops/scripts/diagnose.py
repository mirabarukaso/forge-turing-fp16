"""Report how this Forge venv will load SageAttention, and what to do next.

Run with the virtualenv Python:

    venv/Scripts/python.exe diagnose.py --forge-root <forge>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
import sysconfig
import traceback
from pathlib import Path


def _forge_root(start: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "launch.py").is_file():
            raise SystemExit(f"{root} has no launch.py")
        return root
    for parent in [start, *start.parents]:
        if (parent / "launch.py").is_file() and (parent / "venv").is_dir():
            return parent
    raise SystemExit("Pass --forge-root. This script is not inside a Forge checkout.")


def _dist(*names: str) -> str:
    for name in names:
        try:
            return f"{name} {importlib.metadata.version(name)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return "missing (" + ", ".join(names) + ")"


def _uv_marker(prefix: Path) -> str:
    cfg = prefix / "pyvenv.cfg"
    if not cfg.is_file():
        return "no pyvenv.cfg"
    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "uv":
            return f"uv venv ({value.strip()}), no pip module"
    return "not a uv venv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Sage and the Forge virtualenv.")
    parser.add_argument("--forge-root", type=Path, default=None)
    args = parser.parse_args()

    root = _forge_root(Path(__file__).resolve(), args.forge_root)
    prefix = Path(sys.prefix)
    purelib = Path(sysconfig.get_path("purelib"))
    stub_py = purelib / "turing_sage_stub.py"
    stub_pth = purelib / "turing_sage_stub.pth"
    pip_ok = importlib.util.find_spec("pip") is not None

    print(f"forge: {root}")
    print(f"python: {sys.executable}")
    print(f"prefix: {prefix}")
    print(f"pyvenv: {_uv_marker(prefix)}")
    print(f"pip module: {pip_ok}")
    print("torch:", _dist("torch"))
    print("sageattention:", _dist("sageattention"))
    print("triton:", _dist("triton-windows", "triton"))
    print("xformers:", _dist("xformers"))
    print(f"stub py: {stub_py.is_file()} {stub_py}")
    print(f"stub pth: {stub_pth.is_file()} {stub_pth}")

    sage_ok = False
    fused_dll = False
    try:
        from sageattention import sageattn  # noqa: F401

        sage_ok = True
        print("import sageattention: ok")
    except Exception as exc:
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"import sageattention: {text}")
        fused_dll = "_fused" in text and ("DLL load failed" in text or "specified procedure could not be found" in text or "cannot open shared object" in text.lower())

    if sage_ok:
        print("next: restart-webui-if-log-still-says-xformers")
        return 0
    if fused_dll or not stub_pth.is_file():
        print("next: install-stub")
        return 2
    print("next: read-import-error")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
