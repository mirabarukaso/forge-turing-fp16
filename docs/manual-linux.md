# Linux: manual steps

Run these commands from the Forge Neo root. Replace `/path/to/forge-neo` with that directory. `<` is redirection in bash, so a requirement that contains it must be quoted.

## 1. Virtualenv

```bash
python3 -m venv venv
PY=/path/to/forge-neo/venv/bin/python
```

## 2. PyTorch

Leave an existing torch install alone. If torch is missing, use Forge Neo's `TORCH_COMMAND`. When that variable is unset, match the current default:

```bash
"$PY" -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
```

## 3. Confirm the GPU is Turing

```bash
"$PY" -c "import torch; print(torch.cuda.get_device_capability(0))"
```

The output must be `(7, 5)`. Otherwise stop here.

If the Turing card is not the first visible GPU, `export CUDA_VISIBLE_DEVICES=1` before the probe and the installs.

## 4. Triton

This line makes bash try to read a file named `3.3`:

```bash
"$PY" -m pip install -U --no-deps triton<3.3
```

Quote the requirement:

```bash
"$PY" -m pip install -U --no-deps "triton<3.3"
```

The Linux package name is `triton`, not `triton-windows`. Skip this step when the installed version is already older than 3.3.

uv:

```bash
uv pip install --python "$PY" -U --no-deps "triton<3.3"
```

## 5. SageAttention

Linux does not use the Windows post3 wheel. Install 2.2.0 from PyPI:

```bash
"$PY" -m pip install -U --no-deps "sageattention==2.2.0"
```

Skip this step when `pip show sageattention` is already 2.2.0.

## 6. xformers

Install only when it is missing. Use the same CUDA index as torch:

```bash
"$PY" -m pip install xformers==0.0.35 --extra-index-url https://download.pytorch.org/whl/cu130
```

## 7. Sage import stub

Copy `installer/turing_sage_stub.py` into the virtualenv site-packages directory. The path looks like:

```text
venv/lib/python3.12/site-packages/turing_sage_stub.py
```

Use the Python version from the virtualenv. In that same directory, create `turing_sage_stub.pth`:

```text
import turing_sage_stub
```

The hook checks for sm75 only when SageAttention loads `_fused`. A failed load is replaced with an empty module so the Triton kernels can import. Other GPUs still use the real extension.

## 8. Launch arguments

`webui-user.sh`:

```bash
export COMMANDLINE_ARGS="--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install"
```

`--skip-install` stops the next launch from upgrading Triton to 3.3 or newer, and from replacing Sage with Forge's default newer build. `requirements.txt` and extension `install.py` will not run automatically either. When those are needed:

```bash
"$PY" -m pip install -r requirements.txt
```

Do not add `-U`.

## 9. Extension

Put this repository in `extensions/forge-turing-fp16` and start `webui.sh`. The Optimizations settings page shows the two switches.
