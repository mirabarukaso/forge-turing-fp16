# forge-turing-fp16

Forge Neo extension for Turing (compute capability 7.5). It adds two Optimizations switches and does not modify Forge itself:

- Diffusion compute in fp16
- Cast attention q/k/v to fp16 for SageAttention

Forge's own startup would install a Sage build this GPU cannot use. Keep `--skip-install` in the launch arguments, and install the Turing packages with the script below.

These switches target Turing. Later GPUs may see no speedup, and the fp32 path can keep more weights resident.

## Benchmark

NVIDIA TITAN RTX, 24 GB. SageAttention 2 on this card uses the Triton kernels (`SageAttention CUDA extension unavailable; using Triton kernels`). Times are the sampling loop.

### Forge Neo, latest API check

Model `moss_image_sft_bf16.safetensors`, stored as fp32, with `flux2-vae` and `qwen3.5_9b_int8_convrot`. 8 steps, LoRA `8steps_lora_latest` at 0.7.

| Switches | Sampling | UNet |
| --- | --- | --- |
| Both on | 2.88 s/it (23 s) | 19387 MB, fully loaded |
| Both off | 10.73 s/it (1 m 25 s) | 15907 MB loaded, 3480 MB offloaded, 49 lowvram patches |

About 3.7×. The off state is the same weight set split across GPU and CPU: 15907 + 3480 = 19387 MB. fp32 compute reserves more VRAM, so the loader offloads part of the UNet. It is not a second copy left behind by the fp16 run. `POST /sdapi/v1/unload-checkpoint` between requests does free the previous weights, and `cuda.system.free` rises. The next load still offloads, because Forge sets the weight budget from `computation_dtype` (here 19403 MB usable with fp16, 16221 MB with fp32). An int8 UNet that already fits does not get this split.

### Forge Neo, earlier check

Same card, also 8 steps and the same LoRA. From the closed core PR that this extension replaced.

| Model | Switches off | fp16 + xformers | fp16 + SageAttention 2 |
| --- | --- | --- | --- |
| `moss_image_sft_int8_convrot` (5784 MB) | 5.81 s/it (46 s) | 2.20 s/it (17 s), 2.6× | 1.68 s/it (13 s), 3.5× |
| `moss_image_sft_bf16` (stored as fp32) | 13.85 s/it (1 m 50 s), 3480 MB offloaded | 2.62 s/it (20 s), 5.3× | 2.53 s/it (20 s), 5.5× |

On the int8 model the UNet stays at 5784 MB, so that speedup is not from leaving lowvram. Against xformers with the same fp16 switches, SageAttention 2 was about 1.3× on int8 and about the same on the fp32 model.

`forge_sage_fp16_cast` does not change linear compute. It only runs inside SageAttention. If SageAttention is not the active backend, the switch logs a warning and does nothing.

### Same idea in ComfyUI

Same GPU and the same fp16 compute plus Sage q/k/v cast, measured with [SageFP16Switch](https://github.com/mirabarukaso/ComfyUI_MiraSubPack) rather than this Forge extension.

- MOSS-Image SFT bf16, 1024×1536, 6 steps: about 94 s/image in fp32 default attention → about 20 s/image with the switch and SageAttention 2 (about 4.7×). int8 weights go from about 18 s to about 13 s.
- Qwen-Image 2.1 Q8_0 GGUF, 1024×1024, 25 steps: about 143 s/image native fp32 → about 24–30 s/image in fp16 (about 5–6×). SageAttention 2 is roughly 20% faster than PyTorch attention at the same fp16 compute.

Full tables and LPIPS notes: https://github.com/mirabarukaso/ComfyUI_MiraSubPack#sagefp16switch-benchmark-moss-image-sft-bf16-vs-int8-titan-rtx-sm75

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

An API client can send either key inside `override_settings` on `txt2img`. `override_settings_restore_afterwards: true` restores the saved options after that request. It does not unload weights. Unload only when a switch goes from on to off, or when the checkpoint or additional modules change. The sequence is in [API VRAM](docs/api-vram.md).

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
- [API VRAM](docs/api-vram.md)

Agents that diagnose Sage, uv, or a Forge upgrade should follow [.cursor/skills/forge-turing-ops/SKILL.md](.cursor/skills/forge-turing-ops/SKILL.md).

## After a Forge update, recheck these

The extension wraps these entry points. A signature change, or a module that imports the function by value, makes the wrapper miss:

- `modules.sd_models.forge_model_reload`
- `modules.processing.forge_model_reload` (`processing.py` imports this name at the top of the file)
- `backend.modules.k_model.KModel.apply_model`
- `backend.attention.attention_sage` and `attention.attention_function`
