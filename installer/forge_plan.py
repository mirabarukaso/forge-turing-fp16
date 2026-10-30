"""Read Forge Neo's installer and print the Turing install list.

Forge decides packages in modules/launch_utils.py prepare_environment().
--skip-install makes run_pip() return immediately, so requirements, attention
packages, and extension install.py do not run. Torch still installs through
run() when it is missing or --reinstall-torch is set.

This script does not guess those packages. It reads the current launch_utils.py.
On sm75 it replaces Forge's Sage and Triton specs with the Turing pins.
Without --apply it only prints the list.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

INSTALLER_DIR = Path(__file__).resolve().parent
if str(INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(INSTALLER_DIR))

import turing_install as ti  # noqa: E402

RE_REQUIREMENT = re.compile(r"\s*(\S+)\s*==\s*([^\s;]+)\s*")
RE_MARKER_313 = re.compile(r"""python_version\s*>=\s*['"]3\.13['"]""")


@dataclass
class Item:
    section: str
    name: str
    detail: str
    pip_args: list[str] | None = None
    script: Path | None = None
    apply: bool = False


@dataclass
class Plan:
    notes: list[str] = field(default_factory=list)
    forge: list[Item] = field(default_factory=list)
    selected: list[Item] = field(default_factory=list)


def _must(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"Could not read {label} from modules/launch_utils.py. Update installer/forge_plan.py before installing.")
    return match.group(1)


def _fstrings(env_name: str, text: str) -> list[str]:
    found = re.findall(rf'{re.escape(env_name)}", f"([^"]*)"', text)
    if len(found) != 2:
        raise SystemExit(f"Expected Windows and Linux specs for {env_name} in modules/launch_utils.py, found {len(found)}.")
    return found


def _read_launch_args(forge_root: Path) -> tuple[str, list[str]]:
    if os.name == "nt":
        path = forge_root / "webui-user.bat"
        pattern = r"(?m)^set COMMANDLINE_ARGS=(.*)$"
    else:
        path = forge_root / "webui-user.sh"
        pattern = r'(?m)^export COMMANDLINE_ARGS="(.*)"\s*$'
    if not path.is_file():
        return str(path), []
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if match is None:
        return str(path), []
    raw = match.group(1).strip().strip('"')
    return str(path), raw.split()


def _flag(args: list[str], name: str) -> bool:
    return name in args


def _fill(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _dist(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _older(installed: str, required: str) -> bool:
    try:
        import packaging.version

        return packaging.version.parse(installed) < packaging.version.parse(required)
    except Exception:
        return ti._version_tuple(installed) < ti._version_tuple(required)


def _marker_skips_this_python(line: str) -> bool:
    if ";" not in line:
        return False
    if sys.version_info < (3, 13) and RE_MARKER_313.search(line):
        return True
    return False


def _requirement_gaps(path: Path) -> tuple[list[str], list[str]]:
    gaps: list[str] = []
    ignored: list[str] = []
    if not path.is_file():
        return [f"missing {path}"], ignored
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = RE_REQUIREMENT.match(line)
        if match is None:
            ignored.append(stripped)
            continue
        package, required = match.group(1), match.group(2)
        if _marker_skips_this_python(line):
            ignored.append(f"{stripped} (marker skips this Python)")
            continue
        installed = _dist(package)
        if installed is None:
            gaps.append(f"{package}=={required} is missing")
        elif _older(installed, required):
            gaps.append(f"{package} {installed} is older than {required}")
    return gaps, ignored


def _disabled_names(forge_root: Path) -> set[str]:
    names = {"sd-webui-controlnet", "multidiffusion-upscaler-for-automatic1111"}
    config_py = forge_root / "modules_forge" / "config.py"
    if config_py.is_file():
        text = config_py.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"always_disabled_extensions[^=]*=\s*\[(.*?)\]", text, re.S)
        if block:
            names = set(re.findall(r'"([^"]+)"', block.group(1)))
    settings = forge_root / "config.json"
    if settings.is_file():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if data.get("disable_all_extensions", "none") != "none":
            names.add("*")
        names.update(data.get("disabled_extensions", []))
    return names


def _extension_scripts(forge_root: Path) -> list[Path]:
    disabled = _disabled_names(forge_root)
    if "*" in disabled:
        return []
    found: list[Path] = []
    for folder in ("extensions", "extensions-builtin"):
        root = forge_root / folder
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            script = child / "install.py"
            if child.name in disabled or not script.is_file():
                continue
            found.append(script)
    return found


def _torch_parts() -> tuple[str, str, str] | None:
    version = _dist("torch")
    if version is None:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)(?:[^+]+)?\+(.+)", version)
    if match is None:
        return None
    ver, cuda = match.group(1), match.group(2)
    return ver, cuda, version


def _ensure_venv(forge_root: Path) -> None:
    venv = (forge_root / "venv").resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix != venv:
        raise SystemExit(f"Run this with the Forge virtualenv Python. Current interpreter is {sys.executable}.")


def build_plan(forge_root: Path) -> Plan:
    source = (forge_root / "modules" / "launch_utils.py").read_text(encoding="utf-8", errors="replace")
    args_file, launch_args = _read_launch_args(forge_root)
    plan = Plan()
    skip = _flag(launch_args, "--skip-install")
    plan.notes.append(f"launch args file: {args_file}")
    plan.notes.append("launch args: " + (" ".join(launch_args) if launch_args else "(none)"))
    plan.notes.append(
        "skip-install: run_pip returns immediately, so Sage, Triton, xformers, requirements.txt, and extension install.py do not run."
        if skip
        else "skip-install is absent. A normal startup would install Forge's own Sage and Triton specs."
    )
    plan.notes.append("Torch uses run(), not run_pip(). skip-install does not block a missing torch or --reinstall-torch.")

    torch_index = _must(r'TORCH_INDEX_URL", "([^"]+)"', source, "TORCH_INDEX_URL")
    torch_command = _fill(_must(r'TORCH_COMMAND", f"([^"]+)"', source, "TORCH_COMMAND"), {"torch_index_url": torch_index})
    xformers_spec = _fill(_must(r'XFORMERS_PACKAGE", f"([^"]+)"', source, "XFORMERS_PACKAGE"), {"torch_index_url": torch_index})
    packaging_spec = _must(r'PACKAGING_PACKAGE", "([^"]+)"', source, "PACKAGING_PACKAGE")
    gradio_spec = _must(r'GRADIO_PACKAGE", "([^"]+)"', source, "GRADIO_PACKAGE")
    pynvml_spec = _must(r'PYNVML_PACKAGE", "([^"]+)"', source, "PYNVML_PACKAGE")
    ver_sage = _must(r'ver_SAGE = "([^"]+)"', source, "ver_SAGE")
    ver_flash = _must(r'ver_FLASH = "([^"]+)"', source, "ver_FLASH")
    ver_triton = _must(r'ver_TRITON = "([^"]+)"', source, "ver_TRITON")
    ver_nunchaku = _must(r'ver_NUNCHAKU = "([^"]+)"', source, "ver_NUNCHAKU")

    sage_win, sage_linux = _fstrings("SAGE_PACKAGE", source)
    flash_win, flash_linux = _fstrings("FLASH_PACKAGE", source)
    triton_win, triton_linux = _fstrings("TRITION_PACKAGE", source)
    nunchaku_win, nunchaku_linux = _fstrings("NUNCHAKU_PACKAGE", source)

    values = {
        "ver_SAGE": ver_sage,
        "ver_FLASH": ver_flash,
        "ver_TRITON": ver_triton,
        "ver_NUNCHAKU": ver_nunchaku,
        "ver_PY": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "ver_CUDA": "cu130",
        "v_TORCH": "2.10",
        "v_CUDA": "cu13.0",
    }
    torch_parts = _torch_parts()
    if torch_parts is not None:
        ver, cuda, full = torch_parts
        values["ver_CUDA"] = cuda
        values["v_TORCH"] = ver.rsplit(".", 1)[0]
        values["v_CUDA"] = f"{cuda[:-1]}.{cuda[-1]}" if cuda.startswith("cu") else cuda
        plan.forge.append(Item("leave", "torch", f"installed {full}. Forge's missing-torch command is: {torch_command}"))
    else:
        plan.forge.append(Item("install", "torch", f"missing. Forge would run: {torch_command}"))
        plan.selected.append(Item("install", "torch", torch_command, pip_args=ti._torch_args(), apply=True))

    platform_sage = _fill(sage_win if os.name == "nt" else sage_linux, values)
    platform_triton = _fill(triton_win if os.name == "nt" else triton_linux, values)
    platform_flash = _fill(flash_win if os.name == "nt" else flash_linux, values)
    platform_nunchaku = _fill(nunchaku_win if os.name == "nt" else nunchaku_linux, values)

    capability = ti._capability() if torch_parts is not None else None
    sm75 = capability == (7, 5)
    plan.notes.append(f"gpu capability: {capability}")
    if torch_parts is None:
        plan.notes.append("torch is missing, so the GPU was not probed. Install torch, then run this planner again before choosing Sage pins.")
    elif not sm75:
        plan.notes.append("Capability is not 7.5. Sage and Triton stay on Forge's specs. This planner does not install them.")

    sage_now = ti._dist_version("sageattention")
    triton_now = ti._dist_version("triton-windows", "triton")
    plan.forge.append(Item("forge", "sageattention", f"flag --sage={_flag(launch_args, '--sage')}; spec {platform_sage}; installed {sage_now}"))
    plan.forge.append(Item("forge", "triton", f"installed with Sage; spec {platform_triton}; installed {triton_now}"))

    if _flag(launch_args, "--sage"):
        if sm75:
            if os.name == "nt":
                sage_ok = sage_now is not None and "post3" in sage_now
                sage_detail = "Windows sm75 uses the post3 wheel, not Forge's post6 URL."
            else:
                sage_ok = sage_now is not None and ti._version_tuple(sage_now)[:3] == (2, 2, 0)
                sage_detail = "Linux sm75 uses sageattention==2.2.0."
            triton_ok = triton_now is not None and ti._version_tuple(triton_now) < (3, 3)
            plan.selected.append(Item("keep" if sage_ok else "override", "sageattention", sage_detail + f" Forge spec: {platform_sage}", apply=not sage_ok))
            plan.selected.append(Item("keep" if triton_ok else "override", "triton", f"Turing pin is <3.3. Forge spec: {platform_triton}", apply=not triton_ok))
            stub = Path(sysconfig.get_path("purelib")) / "turing_sage_stub.pth"
            plan.selected.append(Item("keep" if stub.is_file() else "install", "turing_sage_stub", str(stub), apply=not stub.is_file()))

    xformers_now = ti._dist_version("xformers")
    xformers_wanted = _flag(launch_args, "--xformers") or _flag(launch_args, "--reinstall-xformers")
    plan.forge.append(Item("forge", "xformers", f"flag={xformers_wanted}; spec {xformers_spec}; installed {xformers_now}"))
    if xformers_wanted and (xformers_now is None or _flag(launch_args, "--reinstall-xformers")):
        if xformers_now is None:
            plan.selected.append(Item("install", "xformers", xformers_spec, pip_args=xformers_spec.split(), apply=True))
        else:
            plan.selected.append(Item("leave", "xformers", f"installed {xformers_now}. --reinstall-xformers is set; this planner does not upgrade it."))
    elif xformers_wanted and xformers_now is not None:
        plan.selected.append(Item("keep", "xformers", f"installed {xformers_now}. Forge installs {xformers_spec} only when xformers is missing or --reinstall-xformers is set."))

    for name, spec, enabled in (
        ("flash_attn", platform_flash, _flag(launch_args, "--flash")),
        ("nunchaku", platform_nunchaku, _flag(launch_args, "--nunchaku")),
        ("pynvml", pynvml_spec, _flag(launch_args, "--pynvml")),
        ("ngrok", "ngrok", _flag(launch_args, "--ngrok")),
    ):
        installed = _dist(name) if name != "pynvml" else (_dist("nvidia-ml-py") or _dist("pynvml"))
        plan.forge.append(Item("forge", name, f"flag={enabled}; spec {spec}; installed {installed}"))
        if enabled and installed is None:
            plan.selected.append(Item("install", name, spec, pip_args=spec.split(), apply=True))

    for package in packaging_spec.split() + gradio_spec.split():
        if "==" not in package:
            continue
        name, required = package.split("==", 1)
        installed = _dist(name)
        state = "keep"
        apply = False
        if installed is None or _older(installed, required):
            state = "install"
            apply = True
        plan.forge.append(Item(state, name, f"Forge spec {package}; installed {installed}"))
        if apply:
            plan.selected.append(Item("install", name, package, pip_args=[package], apply=True))

    gaps, ignored = _requirement_gaps(forge_root / "requirements.txt")
    if ignored:
        plan.notes.append("requirements.txt lines Forge's checker does not treat as name==version, or this Python skips: " + "; ".join(ignored))
    if gaps:
        detail = "; ".join(gaps)
        plan.forge.append(Item("install", "requirements.txt", detail))
        plan.selected.append(Item("install", "requirements.txt", detail + ". Apply runs one install -r without -U. == pins can move a newer install back to the pin.", pip_args=["-r", str(forge_root / "requirements.txt")], apply=True))
    else:
        plan.forge.append(Item("keep", "requirements.txt", "every name==version pin that applies to this Python is installed and new enough"))

    scripts = _extension_scripts(forge_root)
    if not scripts:
        plan.forge.append(Item("keep", "extension install.py", "none to run"))
    for script in scripts:
        rel = script.relative_to(forge_root).as_posix()
        plan.forge.append(Item("run", rel, "skip-install skips run_extensions_installers"))
        plan.selected.append(Item("run", rel, "run with the venv Python during --apply", script=script, apply=True))

    if _flag(launch_args, "--reinstall-torch"):
        plan.notes.append("--reinstall-torch is set. This planner still leaves an existing torch install in place.")
    return plan


def print_plan(plan: Plan) -> None:
    print("Forge analyzed launch_utils.py prepare_environment().")
    for note in plan.notes:
        print(f"note: {note}")
    print("")
    print("## Forge")
    for item in plan.forge:
        print(f"[{item.section}] {item.name}")
        print(f"  {item.detail}")
    print("")
    print("## Turing install list")
    chosen = [item for item in plan.selected if item.apply]
    if not chosen:
        print("(nothing to install)")
    for item in plan.selected:
        mark = "apply" if item.apply else "satisfied"
        print(f"[{item.section}/{mark}] {item.name}")
        print(f"  {item.detail}")
    print(f"apply-count: {len(chosen)}")


def _run_extension(script: Path, forge_root: Path, use_uv: bool) -> None:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(Path(sys.prefix).resolve())
    env["PYTHONPATH"] = os.pathsep.join([str(forge_root), env.get("PYTHONPATH", "")])
    command = [sys.executable, str(script)]
    if use_uv:
        command.append("--uv")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=str(forge_root))


def apply_plan(plan: Plan, use_uv: bool, forge_root: Path) -> None:
    sm75_done = False
    for item in plan.selected:
        if not item.apply:
            continue
        if item.name in {"sageattention", "triton", "turing_sage_stub"} and not sm75_done:
            ti._install_triton(use_uv)
            ti._install_sage(use_uv)
            ti._install_stub()
            sm75_done = True
            continue
        if item.name in {"sageattention", "triton", "turing_sage_stub"}:
            continue
        if item.script is not None:
            _run_extension(item.script, forge_root, use_uv)
            continue
        if item.pip_args:
            ti._pip(item.pip_args, use_uv)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Turing install list from the current Forge Neo source.")
    parser.add_argument("--forge-root", required=True, type=Path)
    parser.add_argument("--uv", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--device-id", type=int, default=None)
    args = parser.parse_args()

    forge_root = args.forge_root.resolve()
    if not (forge_root / "launch.py").is_file():
        raise SystemExit(f"{forge_root} does not look like a Forge Neo checkout (launch.py is missing).")
    _ensure_venv(forge_root)
    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    plan = build_plan(forge_root)
    print_plan(plan)
    if not args.apply:
        print("next: review the Turing install list. Re-run with --apply to install it.")
        return 0
    use_uv = ti._resolve_use_uv(args.uv)
    apply_plan(plan, use_uv, forge_root)
    print("next: restart WebUI after the user asks.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
