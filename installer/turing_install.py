"""Install Turing pins into a Forge Neo virtualenv.

Run this with the virtualenv's Python. The Windows and Linux shell wrappers do that.
Package specs that contain "<" are passed as arguments, never through a shell.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

INSTALLER_DIR = Path(__file__).resolve().parent
STUB_SOURCE = INSTALLER_DIR / "turing_sage_stub.py"

TORCH_INDEX_URL = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
DEFAULT_TORCH_SPEC = [
    "torch==2.14.0+cu130",
    "torchvision==0.29.0+cu130",
    "--extra-index-url",
    TORCH_INDEX_URL,
]
DEFAULT_XFORMERS = "xformers==0.0.35"

SAGE_TURING_RELEASE = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/"
SAGE_TURING_WHEELS = {
    "cu124": "sageattention-2.2.0+cu124torch2.5.1.post3-cp39-abi3-win_amd64.whl",
    "cu126": "sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl",
    "cu128": "sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl",
    "cu130": "sageattention-2.2.0+cu130torch2.9.0.post3-cp39-abi3-win_amd64.whl",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    core = value.split("+", 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _dist_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return importlib.util.find_spec(name) is not None


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _pip(args: list[str], use_uv: bool) -> None:
    if use_uv:
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit("uv was requested but was not found on PATH. Re-run without --uv.")
        _run([uv, "pip", "install", "--python", sys.executable, *args])
        return
    _run([sys.executable, "-m", "pip", "install", *args])


def _torch_args() -> list[str]:
    raw = os.environ.get("TORCH_COMMAND", "").strip()
    if not raw:
        return list(DEFAULT_TORCH_SPEC)
    parts = raw.split()
    if parts and parts[0] == "pip":
        parts = parts[1:]
    if parts and parts[0] == "install":
        parts = parts[1:]
    return parts


def _capability() -> tuple[int, int] | None:
    import torch

    cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
    if not cuda:
        return None
    return tuple(torch.cuda.get_device_capability(0))


def _cuda_tag() -> str:
    version = importlib.metadata.version("torch")
    match = re.search(r"\+(cu\d+)", version)
    if match is None:
        print(f"Could not read a CUDA tag from torch {version}; using cu130 wheel.")
        return "cu130"
    return match.group(1)


def _install_triton(use_uv: bool) -> None:
    if os.name == "nt":
        package = "triton-windows"
        spec = "triton-windows<3.3"
        names = ("triton-windows", "triton")
    else:
        package = "triton"
        spec = "triton<3.3"
        names = ("triton",)
    current = _dist_version(*names)
    if current is not None and _version_tuple(current) < (3, 3):
        print(f"{package} {current} is already older than 3.3")
        return
    if current is None:
        print(f"{package} is missing")
    else:
        print(f"{package} {current} is 3.3 or newer; replacing it")
    _pip(["install", "-U", "--no-deps", spec], use_uv)


def _install_sage(use_uv: bool) -> None:
    current = _dist_version("sageattention")
    if os.name == "nt":
        tag = _cuda_tag()
        wheel = SAGE_TURING_WHEELS.get(tag, SAGE_TURING_WHEELS["cu130"])
        spec = SAGE_TURING_RELEASE + wheel
        if current is not None and "post3" in current:
            print(f"sageattention {current} is already a post3 build")
            return
        print(f"Installing SageAttention post3 for {tag}")
    else:
        spec = "sageattention==2.2.0"
        if current is not None and _version_tuple(current)[:3] == (2, 2, 0):
            print(f"sageattention {current} is already 2.2.0")
            return
        print("Installing sageattention==2.2.0")
    _pip(["install", "-U", "--no-deps", spec], use_uv)


def _install_xformers(use_uv: bool) -> None:
    if _installed("xformers"):
        print(f"xformers {_dist_version('xformers')} is already installed")
        return
    index = os.environ.get("TORCH_INDEX_URL", f"https://download.pytorch.org/whl/{_cuda_tag()}")
    _pip(["install", DEFAULT_XFORMERS, "--extra-index-url", index], use_uv)


def _install_requirements(forge_root: Path, use_uv: bool) -> None:
    requirements = forge_root / "requirements.txt"
    if not requirements.is_file():
        raise SystemExit(f"requirements.txt was not found at {requirements}")
    _pip(["install", "-r", str(requirements)], use_uv)


def _install_stub() -> None:
    if not STUB_SOURCE.is_file():
        raise SystemExit(f"Missing stub source: {STUB_SOURCE}")
    site = Path(sysconfig.get_path("purelib"))
    site.mkdir(parents=True, exist_ok=True)
    target = site / "turing_sage_stub.py"
    pth = site / "turing_sage_stub.pth"
    shutil.copyfile(STUB_SOURCE, target)
    pth.write_text("import turing_sage_stub\n", encoding="utf-8")
    print(f"Wrote {target}")
    print(f"Wrote {pth}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pin SageAttention and Triton for a Turing Forge Neo venv.")
    parser.add_argument("--forge-root", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--uv", action="store_true")
    parser.add_argument("--requirements", action="store_true")
    args = parser.parse_args()

    forge_root = args.forge_root.resolve()
    if not (forge_root / "launch.py").is_file():
        raise SystemExit(f"{forge_root} does not look like a Forge Neo checkout (launch.py is missing).")

    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        print(f"CUDA_VISIBLE_DEVICES={args.device_id}")

    if not _installed("torch"):
        print("torch is missing; installing the Forge Neo default build")
        _pip(["install", *_torch_args()], args.uv)

    capability = _capability()
    if capability != (7, 5):
        print(f"This installer only changes Sage, Triton, xformers, and the import stub on sm75. Detected {capability}.")
        return 1

    _install_triton(args.uv)
    _install_sage(args.uv)
    _install_xformers(args.uv)
    _install_stub()
    if args.requirements:
        _install_requirements(forge_root, args.uv)

    print("sageattention", _dist_version("sageattention"))
    print("triton", _dist_version("triton-windows", "triton"))
    print("xformers", _dist_version("xformers"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
