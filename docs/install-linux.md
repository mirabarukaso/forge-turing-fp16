# Linux: install with the script

Use this after the Forge Neo directory can create a `venv`. The script passes arguments as a list to pip or uv, so `triton<3.3` is not treated as bash redirection.

## 1. Place the extension

Use Forge's Install from URL. The repository URL is `https://github.com/mirabarukaso/forge-turing-fp16`. Leave the branch and the local directory name empty. Forge clones it to `extensions/forge-turing-fp16`. Apply and restart from the Installed tab. This does not install the Turing packages. Continue with the steps below.

To clone by hand, from the Forge Neo root:

```bash
git clone https://github.com/mirabarukaso/forge-turing-fp16.git extensions/forge-turing-fp16
chmod +x extensions/forge-turing-fp16/installer/install-turing.sh extensions/forge-turing-fp16/installer/upgrade-forge.sh
```

Or copy this repository to `/path/to/forge-neo/extensions/forge-turing-fp16`, then `chmod +x` the two shell scripts. A copy without a `.git` directory cannot be updated from the Extensions tab.

## 2. Turn off automatic WebUI installs

Edit `webui-user.sh`. A Turing setup can use:

```bash
export COMMANDLINE_ARGS="--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install"
```

Do not add `--reinstall-torch`. `--skip-install` skips Sage, Triton, xformers, `requirements.txt`, and extension `install.py`. Forge still installs torch when torch is missing, or when `--reinstall-torch` is set.

After a Forge Neo upgrade, `requirements.txt` and extension `install.py` will not run automatically either. Use [the upgrade steps](upgrade.md). `--requirements` on this installer only installs `requirements.txt`. It does not run extension `install.py`.

## 3. Run the script

```bash
./installer/install-turing.sh --forge-root /path/to/forge-neo --uv
```

`--forge-root` can be omitted when the extension already lives in `extensions/forge-turing-fp16`. Pass `--uv` when the WebUI is started with `--uv`, or when the virtualenv should be created with `uv venv`. Omit `--uv` when creating a normal virtualenv and uv is not installed.

If `venv/pyvenv.cfg` already contains a line `uv =`, the installer uses `uv pip install --python <venv python>` even without `--uv`. That virtualenv has no pip module. Forge's `--uv` flag does not create the virtualenv. It rewrites `pip` into `uv pip` after `webui.sh` activates the virtualenv. This installer always passes `--python`.

Optional arguments:

- `--device-id 1` sets `CUDA_VISIBLE_DEVICES` before the probe.
- `--requirements` runs `pip install -r requirements.txt` without `-U`.

If the detected compute capability is not 7.5, the script exits and does not change Sage, Triton, xformers, or the import stub. On Linux, Sage is the PyPI package `sageattention==2.2.0`. The Triton specifier is `triton<3.3`, not `triton-windows`.

## 4. Check

```bash
/path/to/forge-neo/venv/bin/python -c "import importlib.metadata as m; print(m.version('sageattention')); print(m.version('triton')); print(m.version('xformers'))"
```

Do not use `python -m pip show` on a virtualenv whose `pyvenv.cfg` contains `uv =`.

Expected:

- `sageattention` is 2.2.0.
- `triton` is older than 3.3.
- `venv/lib/pythonX.Y/site-packages/turing_sage_stub.pth` exists.

Start `webui.sh`. If WebUI was already running, restart it. The stub is loaded only when Python starts. A Triton fallback logs both of these lines:

```text
SageAttention CUDA extension unavailable; using Triton kernels
Using SageAttention 2
```

If the CUDA extension itself loads, the first line is absent and `Using SageAttention 2` is still the success line. `Using xformers Cross Attention` means Sage did not import.
