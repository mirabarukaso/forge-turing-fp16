# Upgrade Forge Neo while --skip-install stays on

`--skip-install` makes `run_pip()` return immediately. A git update then does not install new `requirements.txt` pins, attention packages, or extension `install.py`. Torch is the exception: Forge installs it with `run()`, so a missing torch or `--reinstall-torch` still runs.

Removing `--skip-install` to let startup finish the upgrade also installs Forge's own Sage and Triton specs. On Windows that is the post6 wheel and `triton-windows<3.9`. Turing needs the post3 wheel and `triton-windows<3.3`.

Keep `--skip-install`. Upgrade with this planner instead.

## What the planner reads

`installer/forge_plan.py` reads the current `modules/launch_utils.py` `prepare_environment()`, `requirements.txt`, `webui-user.bat` or `webui-user.sh`, and extension `install.py` files. It does not keep a second copy of Forge's package URLs. If Forge changes that function so the parser cannot find a line, the script stops. Update the parser before installing anything.

It then prints two sections:

- **Forge**: what that checkout would install for the current launch arguments.
- **Turing install list**: the subset to apply. Sage and Triton are replaced on sm75. An existing torch install is left in place. xformers is installed only when it is missing.

Without `--apply` / `-Apply` nothing is installed. Close WebUI before applying. Start it again after the command finishes. The running process keeps the old Sage import.

## Windows

Print the list:

```powershell
.\installer\upgrade-forge.ps1 -ForgeRoot F:\sd-webui-forge-neo
```

Update the Forge git checkout, then print the list. This is `git pull --ff-only`. Local edits to a file that the pull also changes will stop the pull. `webui-user.bat` is one of those files.

```powershell
.\installer\upgrade-forge.ps1 -ForgeRoot F:\sd-webui-forge-neo -Pull
```

Install the Turing list. A uv virtualenv is detected by `forge_plan.py`; `-UseUv` forces it.

```powershell
.\installer\upgrade-forge.ps1 -ForgeRoot F:\sd-webui-forge-neo -Apply
```

## Linux

```bash
chmod +x installer/upgrade-forge.sh
./installer/upgrade-forge.sh --forge-root /path/to/forge-neo
./installer/upgrade-forge.sh --forge-root /path/to/forge-neo --pull
./installer/upgrade-forge.sh --forge-root /path/to/forge-neo --apply
```

`-Apply` / `--apply` can install `packaging` and other Forge pins that are older than `launch_utils.py` asks for, and it runs each enabled extension `install.py`. Sage, Triton, xformers, and the import stub are included only when the Turing list says they are not already satisfied. An existing torch install is left in place.

## After a code update

`git pull` of Forge Neo does not update `extensions/`. That directory is gitignored. When this extension was installed from URL or with `git clone`, update it from the Installed tab with Check for updates. Copy the files again only when `extensions/forge-turing-fp16` has no `.git` directory.

Recheck the wrapped functions listed in the README. A signature change in Forge makes the monkeypatch miss.
