# Windows: manual steps

Run these commands from the Forge Neo root. Replace `F:\sd-webui-forge-neo` with that directory. Pass the package name as its own argument. Do not paste a requirement that contains `<` into one unquoted `cmd /c` string.

## 1. Virtualenv

If `venv\Scripts\python.exe` does not exist yet:

```bat
python -m venv venv
```

The rest of this page uses:

```bat
set PY=F:\sd-webui-forge-neo\venv\Scripts\python.exe
```

## 2. PyTorch

Leave an existing torch install alone. If torch is missing, match the current Forge Neo default:

```bat
"%PY%" -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
```

If `TORCH_COMMAND` is set, use that command instead of the line above.

## 3. Confirm the GPU is Turing

```bat
"%PY%" -c "import torch; print(torch.cuda.get_device_capability(0))"
```

The output must be `(7, 5)`. Otherwise stop here and do not change Sage or Triton.

If the Turing card is not the first visible GPU, run `set CUDA_VISIBLE_DEVICES=1` in the same terminal (use the real index), then probe and install.

## 4. Triton

This line is wrong in **cmd.exe**:

```bat
"%PY%" -m pip install -U --no-deps triton-windows<3.3
```

`<3.3` is read as “take stdin from a file named `3.3`”, and cmd reports `The system cannot find the file specified.`

Quote the requirement:

```bat
"%PY%" -m pip install -U --no-deps "triton-windows<3.3"
```

Skip this step when `triton-windows` is already installed and older than 3.3.

uv needs the same quotes:

```bat
uv pip install --python "%PY%" -U --no-deps "triton-windows<3.3"
```

## 5. SageAttention 2.2 post3

Pick the wheel that matches the torch CUDA build. Use the last URL for `+cu130`.

```text
cu124  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu124torch2.5.1.post3-cp39-abi3-win_amd64.whl
cu126  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl
cu128  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl
cu130  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu130torch2.9.0.post3-cp39-abi3-win_amd64.whl
```

```bat
"%PY%" -m pip install -U --no-deps https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu130torch2.9.0.post3-cp39-abi3-win_amd64.whl
```

Skip this step when `pip show sageattention` already contains `post3`.

## 6. xformers

Install only when it is missing. Use the same CUDA index as torch:

```bat
"%PY%" -m pip install xformers==0.0.35 --extra-index-url https://download.pytorch.org/whl/cu130
```

## 7. Sage import stub

Copy `installer\turing_sage_stub.py` from this repository to:

```text
venv\Lib\site-packages\turing_sage_stub.py
```

In that same directory, create `turing_sage_stub.pth` with one line:

```text
import turing_sage_stub
```

Python imports that file at startup and installs the hook. The hook does not import torch at startup. It checks for sm75 only when SageAttention loads `_fused`. A failed load is replaced with an empty module so the fp16 Triton kernels can import. Other GPUs still use the real extension.

## 8. Launch arguments

`webui-user.bat`:

```bat
set COMMANDLINE_ARGS=--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install
```

`--skip-install` stops the next launch from upgrading Sage to post6 and Triton to 3.3 or newer. It also skips `requirements.txt` and extension `install.py`. When those are needed:

```bat
"%PY%" -m pip install -r requirements.txt
```

Do not add `-U`. `requirements.txt` does not pin sage or triton.

## 9. Extension

Put this repository in `extensions\forge-turing-fp16` and start `webui.bat`. The Optimizations settings page shows the two switches.
