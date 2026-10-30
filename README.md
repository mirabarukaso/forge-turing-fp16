# forge-turing-fp16

Forge Neo extension for Turing (compute capability 7.5). It adds two Optimizations switches and does not modify Forge itself:

- Diffusion compute in fp16
- Cast attention q/k/v to fp16 for SageAttention

Forge's own startup would install a Sage build this GPU cannot use. Keep `--skip-install` in the launch arguments, and install the Turing packages with the script below.

## What to do

### 1. Put the extension in Forge

Use Forge's **Install from URL**. Open the Extensions tab, then Install from URL:

- URL for extension's git repository: `https://github.com/mirabarukaso/forge-turing-fp16`
- Specific branch name: leave empty. The default branch is `main`.
- Local directory name: leave empty. Forge creates `extensions/forge-turing-fp16`.

Click Install. On the Installed tab, apply and restart. That clones the repository. It does not install the Turing Sage and Triton packages. Do that in step 3.

The same directory can be created by hand. From the Forge Neo root:

```bash
git clone https://github.com/mirabarukaso/forge-turing-fp16.git extensions/forge-turing-fp16
```

Or copy this repository to `<Forge Neo>/extensions/forge-turing-fp16`. A copy without a `.git` directory cannot be updated from the Extensions tab. Install from URL and `git clone` can.

### 2. Change the launch arguments

Windows, `webui-user.bat`:

```bat
set COMMANDLINE_ARGS=--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install
```

Linux, `webui-user.sh`:

```bash
export COMMANDLINE_ARGS="--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install"
```

Keep whatever extra arguments you already use, and add `--sage` and `--skip-install` if they are missing. Do not add `--reinstall-torch`. Do not remove `--skip-install` later. Without it, the next startup installs Forge's Sage wheel (Windows post6) and a Triton newer than 3.3.

`--skip-install` also skips `requirements.txt` and extension `install.py`. Torch is separate: Forge still installs it when torch is missing. Upgrading Forge is step 6, not a reason to remove `--skip-install`.

### 3. Install the Turing packages

The script reads the virtualenv. If `venv/pyvenv.cfg` contains `uv =`, that environment has no pip module and the script uses `uv pip install --python`. You do not have to pass `--uv` for an existing uv virtualenv.

Windows, PowerShell, from this repository:

```powershell
.\installer\install-turing.ps1 -ForgeRoot F:\sd-webui-forge-neo
```

Linux:

```bash
chmod +x installer/install-turing.sh installer/upgrade-forge.sh
./installer/install-turing.sh --forge-root /path/to/forge-neo
```

`-ForgeRoot` / `--forge-root` can be omitted when this repository is already in `extensions/forge-turing-fp16`. If the GPU is not compute capability 7.5, the script stops and does not change Sage, Triton, xformers, or the import stub.

### 4. Start Forge and read the log

Start `webui.bat` or `webui.sh`. If Forge was already running while you installed the stub, restart it. The new process should log:

```text
SageAttention CUDA extension unavailable; using Triton kernels
Using SageAttention 2
Forge Turing FP16 patches installed
```

The first line is absent when the Sage CUDA extension loads on its own. VAE attention can still say xformers. `Using xformers Cross Attention` means Sage did not import. Run the installer again and restart.

### 5. Turn the switches on

In the WebUI: Settings, Optimizations.

- **Diffusion compute in fp16** stores the choice as `forge_compute_fp16`.
- **Cast attention q/k/v to fp16 for SageAttention** stores the choice as `forge_sage_fp16_cast`.

Both start off. Turning one on patches the model that is already loaded. Turning it off does not convert that model back. The next generation reloads the checkpoint.

An API client can send either key inside `override_settings` on `txt2img` or `img2img`. `override_settings_restore_afterwards: true` restores the saved options after that request. It does not unload weights. A frontend that changes the checkpoint or turns these switches off should `POST /sdapi/v1/unload-checkpoint` after the previous job is idle. The sequence is in [the frontend note](.cursor/skills/forge-turing-ops/vram-api.md).

### 6. Upgrade Forge later

Leave `--skip-install` in place. From this repository:

```powershell
.\installer\upgrade-forge.ps1 -ForgeRoot F:\sd-webui-forge-neo
.\installer\upgrade-forge.ps1 -ForgeRoot F:\sd-webui-forge-neo -Pull
.\installer\upgrade-forge.ps1 -ForgeRoot F:\sd-webui-forge-neo -Apply
```

```bash
./installer/upgrade-forge.sh --forge-root /path/to/forge-neo
./installer/upgrade-forge.sh --forge-root /path/to/forge-neo --pull
./installer/upgrade-forge.sh --forge-root /path/to/forge-neo --apply
```

The first command only prints what Forge would install and what this Turing setup should install. `-Pull` / `--pull` is `git pull --ff-only` of Forge Neo itself. `-Apply` / `--apply` installs that list. Close WebUI before applying, then start it again. Forge's pull does not update `extensions/`. Update this extension from the Installed tab with Check for updates when it was installed from the URL or with `git clone`. Copy the files again only when the folder has no `.git` directory.

Details: [Upgrade Forge Neo](docs/upgrade.md).

## If you need the long version

- [Windows installer](docs/install-windows.md)
- [Linux installer](docs/install-linux.md)
- [Windows manual steps](docs/manual-windows.md)
- [Linux manual steps](docs/manual-linux.md)
- [Windows tests](docs/testing-windows.md)
- [Linux tests](docs/testing-linux.md)
- [Upgrade](docs/upgrade.md)

Agents that diagnose Sage, uv, or a Forge upgrade should follow [.cursor/skills/forge-turing-ops/SKILL.md](.cursor/skills/forge-turing-ops/SKILL.md).

## After a Forge update, recheck these

The extension wraps these entry points. A signature change, or a module that imports the function by value, makes the wrapper miss:

- `modules.sd_models.forge_model_reload`
- `modules.processing.forge_model_reload` (`processing.py` imports this name at the top of the file)
- `backend.modules.k_model.KModel.apply_model`
- `backend.attention.attention_sage` and `attention.attention_function`
