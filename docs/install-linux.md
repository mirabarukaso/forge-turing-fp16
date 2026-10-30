# Linux: install with the script

Use this after the Forge Neo directory can create a `venv`. The script passes arguments as a list to pip or uv, so `triton<3.3` is not treated as bash redirection.

## 1. Place the extension

```bash
cd /path/to/forge-neo/extensions
git clone https://github.com/mirabarukaso/forge-turing-fp16.git
chmod +x forge-turing-fp16/installer/install-turing.sh
```

## 2. Turn off automatic WebUI installs

Edit `webui-user.sh`. A Turing setup can use:

```bash
export COMMANDLINE_ARGS="--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install"
```

Do not add `--reinstall-torch`. `--skip-install` skips Sage, Triton, xformers, `requirements.txt`, and extension `install.py`. Forge still installs torch when torch is missing, or when `--reinstall-torch` is set.

After a Forge Neo upgrade, pass `--requirements` to this script when dependencies need to be installed.

## 3. Run the script

```bash
./installer/install-turing.sh --forge-root /path/to/forge-neo --uv
```

`--forge-root` can be omitted when the extension already lives in `extensions/forge-turing-fp16`. Omit `--uv` when uv is not installed.

Optional arguments:

- `--device-id 1` sets `CUDA_VISIBLE_DEVICES` before the probe.
- `--requirements` runs `pip install -r requirements.txt` without `-U`.

If the detected compute capability is not 7.5, the script exits and does not change Sage, Triton, xformers, or the import stub. On Linux, Sage is the PyPI package `sageattention==2.2.0`. The Triton specifier is `triton<3.3`, not `triton-windows`.

## 4. Check

```bash
/path/to/forge-neo/venv/bin/python -m pip show sageattention triton xformers
```

Expected:

- `sageattention` is 2.2.0.
- `triton` is older than 3.3.
- `venv/lib/pythonX.Y/site-packages/turing_sage_stub.pth` exists.

Start `webui.sh`. When the CUDA extension fails to load and Triton is used instead, the log contains:

```text
SageAttention CUDA extension unavailable; using Triton kernels
```
