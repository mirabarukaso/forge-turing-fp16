# Windows: install with the script

Use this after the Forge Neo directory can create a `venv`. The script does not go through `cmd.exe`, so a `<` in a version specifier is not treated as redirection.

## 1. Place the extension

Use Forge's Install from URL. The repository URL is `https://github.com/mirabarukaso/forge-turing-fp16`. Leave the branch and the local directory name empty. Forge clones it to `extensions\forge-turing-fp16`. Apply and restart from the Installed tab. This does not install the Turing packages. Continue with the steps below.

To clone by hand, from the Forge Neo root:

```bat
git clone https://github.com/mirabarukaso/forge-turing-fp16.git extensions\forge-turing-fp16
```

Or copy this repository to `<Forge Neo>\extensions\forge-turing-fp16`. A copy without a `.git` directory cannot be updated from the Extensions tab.

## 2. Turn off automatic WebUI installs

Edit `webui-user.bat`. Keep the existing arguments and add `--skip-install`. One working line is:

```bat
set COMMANDLINE_ARGS=--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install
```

Do not add `--reinstall-torch`. `--skip-install` skips Sage, Triton, xformers, `requirements.txt`, and extension `install.py`. It does not stop PyTorch: Forge still installs torch when torch is missing, or when `--reinstall-torch` is set.

After a Forge Neo upgrade, `requirements.txt` and extension `install.py` will not run automatically either. Use [the upgrade steps](upgrade.md). `-Requirements` on this installer only installs `requirements.txt`. It does not run extension `install.py`.

## 3. Run the script

In PowerShell, from this repository:

```powershell
.\installer\install-turing.ps1 -ForgeRoot F:\sd-webui-forge-neo -UseUv
```

`-ForgeRoot` can be omitted when the extension already lives in `extensions\forge-turing-fp16`. Pass `-UseUv` when the WebUI is started with `--uv`, or when the virtualenv should be created with `uv venv`.

If `venv\pyvenv.cfg` already contains a line `uv =`, the installer uses `uv pip install --python <venv python>` even without `-UseUv`. That virtualenv has no pip module, so `python -m pip` fails. Forge's own `--uv` flag does not create the virtualenv. `webui.bat` still runs `python -m venv` when the folder is missing, then `activate.bat`. During startup, `--uv` rewrites a `pip` command into `uv pip` and relies on `VIRTUAL_ENV` from the activate script. This installer runs outside that process and always passes `--python`.

Optional arguments:

- `-DeviceId 1` sets `CUDA_VISIBLE_DEVICES` before the capability probe. Use it when the Turing card is not the first visible GPU.
- `-Requirements` runs `pip install -r requirements.txt` without `-U`, so the pinned Sage and Triton packages stay in place.

If the detected compute capability is not 7.5, the script exits and does not change Sage, Triton, xformers, or the import stub. If torch was missing, the script installs the current Forge Neo default torch first, because the probe needs it.

## 4. Check

```powershell
F:\sd-webui-forge-neo\venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('sageattention')); print(m.version('triton-windows')); print(m.version('xformers'))"
```

Do not use `python -m pip show` on a virtualenv whose `pyvenv.cfg` contains `uv =`.

Expected:

- The `sageattention` version string contains `post3`.
- `triton-windows` is older than 3.3.
- `venv\Lib\site-packages\turing_sage_stub.pth` exists.

Then start `webui.bat`. If WebUI was already running, restart it. The stub is loaded only when Python starts. A Triton fallback logs both of these lines:

```text
SageAttention CUDA extension unavailable; using Triton kernels
Using SageAttention 2
```

If the CUDA extension itself loads, the first line is absent and `Using SageAttention 2` is still the success line. `Using xformers Cross Attention` means Sage did not import.
