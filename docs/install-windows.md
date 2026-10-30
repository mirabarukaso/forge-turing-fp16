# Windows: install with the script

Use this after the Forge Neo directory can create a `venv`. The script does not go through `cmd.exe`, so a `<` in a version specifier is not treated as redirection.

## 1. Place the extension

Put this repository at:

```text
<Forge Neo>\extensions\forge-turing-fp16
```

## 2. Turn off automatic WebUI installs

Edit `webui-user.bat`. Keep the existing arguments and add `--skip-install`. One working line is:

```bat
set COMMANDLINE_ARGS=--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install
```

Do not add `--reinstall-torch`. `--skip-install` skips Sage, Triton, xformers, `requirements.txt`, and extension `install.py`. It does not stop PyTorch: Forge still installs torch when torch is missing, or when `--reinstall-torch` is set.

After a Forge Neo upgrade, `requirements.txt` and extension `install.py` will not run automatically either. Pass `-Requirements` to this script when those need to be installed.

## 3. Run the script

In PowerShell, from this repository:

```powershell
.\installer\install-turing.ps1 -ForgeRoot F:\sd-webui-forge-neo -UseUv
```

`-ForgeRoot` can be omitted when the extension already lives in `extensions\forge-turing-fp16`. Pass `-UseUv` when the WebUI is started with `--uv`. Without uv, omit the switch and the script uses pip from the virtualenv.

Optional arguments:

- `-DeviceId 1` sets `CUDA_VISIBLE_DEVICES` before the capability probe. Use it when the Turing card is not the first visible GPU.
- `-Requirements` runs `pip install -r requirements.txt` without `-U`, so the pinned Sage and Triton packages stay in place.

If the detected compute capability is not 7.5, the script exits and does not change Sage, Triton, xformers, or the import stub. If torch was missing, the script installs the current Forge Neo default torch first, because the probe needs it.

## 4. Check

```powershell
F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pip show sageattention triton-windows xformers
```

Expected:

- The `sageattention` version string contains `post3`.
- `triton-windows` is older than 3.3.
- `venv\Lib\site-packages\turing_sage_stub.pth` exists.

Then start `webui.bat`. When Sage falls back to Triton, the log contains:

```text
SageAttention CUDA extension unavailable; using Triton kernels
```

If the CUDA extension itself loads, that line is absent. That is expected.
