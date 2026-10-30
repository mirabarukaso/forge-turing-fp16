---
name: forge-turing-ops
description: >-
  Diagnose and fix Forge Neo Turing FP16 extension failures: SageAttention
  staying on xformers, sageattention._fused DLL load errors, uv virtualenvs
  with no pip module, installer pins, and VRAM after fp16 or checkpoint
  switches. Use when Sage does not activate, python -m pip fails in the Forge
  venv, override_settings reloads a model, the user asks for unload-checkpoint
  or a frontend VRAM flow, or Forge Neo needs an upgrade while --skip-install
  is set.
---

# Forge Turing ops

The extension source of truth is the `forge-turing-fp16` repository. A copy under Forge `extensions/forge-turing-fp16` can be stale. Edit the source repository, then copy it into `extensions/forge-turing-fp16`, excluding `.git` and `__pycache__`. Do not commit Forge Neo. Do not start or restart WebUI unless the user asks. Do not upgrade torch when it is already installed. Do not remove `--skip-install`.

## First step

Run the diagnose script with the virtualenv Python. Read its stdout before installing or editing.

Windows:

```text
<forge>/venv/Scripts/python.exe <extension>/.cursor/skills/forge-turing-ops/scripts/diagnose.py --forge-root <forge>
```

Linux:

```text
<forge>/venv/bin/python <extension>/.cursor/skills/forge-turing-ops/scripts/diagnose.py --forge-root <forge>
```

Apply the `next:` line from that output. Do not invent a second install path.

## Sage stays on xformers

`--sage` only asks `modules/launch_utils.py` to install a package. Runtime selection is `backend/memory_management.py` `sage_enabled()`: NVIDIA GPU and `from sageattention import sageattn` succeeded, and `--disable-sage` is absent. `--sage` does not override a failed import. Attention order is Comfy-Kitchen int8, then Sage, then Flash, then xformers. VAE can still log xformers.

A Windows post3 wheel is built for an older torch. Import then fails inside `sageattention/quant.py` with `DLL load failed while importing _fused`. The fix on sm75 is the site-packages stub, not Forge's post6 wheel. `--skip-install` makes `run_pip` return immediately, so WebUI will not install the stub.

When diagnose says `next: install-stub`, run `installer/install-turing.ps1` or `installer/install-turing.sh` against that Forge root. The installer skips Sage, Triton, and xformers when the pins are already satisfied, and it writes `turing_sage_stub.py` plus `turing_sage_stub.pth`. A `.pth` file loads on the next interpreter start. The running WebUI keeps the old import until the user restarts it.

After restart, a Triton fallback logs `SageAttention CUDA extension unavailable; using Triton kernels` and then `Using SageAttention 2`. If the real `_fused` loads, the first line is absent. That is success too.

## uv virtualenv

`venv/pyvenv.cfg` with a line `uv =` means `uv venv` created it. That environment has no pip module. `python -m pip` fails. List packages with `importlib.metadata` or `uv pip list --python <venv python>`.

Forge `webui.bat` / `webui.sh` still create a missing venv with `python -m venv`. `--uv` does not. `launch.py` patches `subprocess.run`: a command containing `pip` becomes `uv pip` plus the arguments after `pip`. It drops `--prefer-binary`, `--ignore-installed`, and `-I`. It does not pass `--python`. That works only because activate set `VIRTUAL_ENV`.

Outside that process, install with `uv pip install --python <venv python>`. Keep `<` specifiers as their own argument. Do not pass them through an unquoted `cmd.exe` string. `triton-windows<3.3` is redirection there.

`installer/turing_install.py` selects uv by itself when `pyvenv.cfg` has `uv` or `pip` cannot be imported. `-UseUv` / `--uv` forces uv and, when the venv folder is missing, creates it with `uv venv`.

## VRAM after a switch

Do not add a plugin flag that calls unload again. `forge_model_reload` already calls `unload_all_models()` and `soft_empty_cache()`. The smaller weight budget after fp16 turns off comes from `KModel.memory_required`, which scales with `computation_dtype`. Another empty_cache does not change that reservation.

The frontend can drop weights left between requests. Read [docs/api-vram.md](../../../docs/api-vram.md) and give that sequence to the frontend. Do not add a new unload endpoint in this extension. Compare the last request that was sent, not the options restored afterward. Unload when a switch goes from on to off, or when the checkpoint or additional modules change. Skip a switch that goes from off to on.

## Upgrade Forge Neo

Do not remove `--skip-install`. Startup would then install Forge's Sage and Triton specs from `prepare_environment` (Windows post6 and `triton-windows<3.9`). Torch is unaffected by `--skip-install` because it uses `run()`, not `run_pip()`.

When the user asks to upgrade Forge, read [docs/upgrade.md](../../../docs/upgrade.md) and run the upgrade script. A run without `-Apply` / `--apply` only prints the list. Do not pass `-Pull` or `-Apply` unless the user asked for that step.

```text
installer/upgrade-forge.ps1 -ForgeRoot <forge>
installer/upgrade-forge.sh --forge-root <forge>
```

The planner reads the current `modules/launch_utils.py`. If it exits because a line could not be parsed, update `installer/forge_plan.py` from that function. Do not type package URLs from memory.

`git pull --ff-only` does not update gitignored `extensions/`. Copy this extension again after the source changes, and recheck the wrapped symbols in the README. Do not restart WebUI unless the user asks.

## After changing the extension

Copy the source tree into Forge `extensions/forge-turing-fp16`. Leave Forge's git status alone except files the user already asked to change.
