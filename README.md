# forge-turing-fp16

Forge Neo 扩展。在不修改 Forge Neo 本体的情况下，给 Turing（sm75）提供两项开关：

- Diffusion compute in fp16
- Cast attention q/k/v to fp16 for SageAttention

SageAttention 2.2 post3 和 `triton<3.3` 不会由 WebUI 自动安装。用本仓库的安装脚本或手动步骤写入虚拟环境，并在启动参数里加上 `--skip-install`，避免下次启动把包装升回 post6 / Triton 3.3+。

## 装到哪里

别人从 GitHub 安装时，把本仓库克隆到 Forge Neo 的扩展目录：

```text
extensions/forge-turing-fp16
```

这台调试机器的仓库在 `W:\Github\forge-turing-fp16`，并用目录联接挂到当前工作区的 `extensions\forge-turing-fp16`。联接只存在于这台机器。

## 文档

- [Windows 脚本安装](docs/install-windows.md)
- [Linux 脚本安装](docs/install-linux.md)
- [Windows 手动步骤](docs/manual-windows.md)
- [Linux 手动步骤](docs/manual-linux.md)
- [Windows 测试](docs/testing-windows.md)
- [Linux 测试](docs/testing-linux.md)

## 跟上 Forge Neo 之后要核对的符号

扩展包装了这些入口。上游改了签名或改成按值导入时，包装会失效：

- `modules.sd_models.forge_model_reload`
- `modules.processing.forge_model_reload`（`processing.py` 在文件开头按名字导入）
- `backend.modules.k_model.KModel.apply_model`
- `backend.attention.attention_sage` 和 `attention.attention_function`
