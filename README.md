# forge-turing-fp16

Forge Neo extension. It adds two switches for Turing (sm75) without modifying Forge Neo itself:

- Diffusion compute in fp16
- Cast attention q/k/v to fp16 for SageAttention

The WebUI does not install SageAttention 2.2 post3 or `triton<3.3`. Use the installer in this repository, or the manual steps, to put those packages in the virtualenv. Add `--skip-install` to the launch arguments so the next startup does not upgrade them to post6 or Triton 3.3+.

## Where to install

Clone this repository into the Forge Neo extensions directory:

```text
extensions/forge-turing-fp16
```

## Documentation

- [Windows installer](docs/install-windows.md)
- [Linux installer](docs/install-linux.md)
- [Windows manual steps](docs/manual-windows.md)
- [Linux manual steps](docs/manual-linux.md)
- [Windows tests](docs/testing-windows.md)
- [Linux tests](docs/testing-linux.md)

## Symbols to recheck after a Forge Neo update

The extension wraps these entry points. The wrappers stop matching upstream if a signature changes, or if a module imports the function by value:

- `modules.sd_models.forge_model_reload`
- `modules.processing.forge_model_reload` (`processing.py` imports this name at the top of the file)
- `backend.modules.k_model.KModel.apply_model`
- `backend.attention.attention_sage` and `attention.attention_function`
