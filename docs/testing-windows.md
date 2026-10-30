# Windows：测试

在 Turing（sm75）上手工跑。这里的清单不能代替出图。

启动前确认 `webui-user.bat` 含有 `--sage --skip-install`，并且没有 `--reinstall-torch`。用 `webui.bat` 启动。

## 包装

```bat
venv\Scripts\python.exe -m pip show sageattention triton-windows
```

- `sageattention` 含 `post3`，重启 WebUI 后仍不是 post6。
- `triton-windows` 小于 3.3。
- CUDA 扩展加载失败时，启动日志有 `using Triton kernels`。扩展能加载时没有这行。

不用 uv、在 cmd 里执行手动文档中的 Triton 安装时，命令带引号，不再出现找不到文件 `3.3`。

## 开关

设置在 Optimizations：

- Diffusion compute in fp16
- Cast attention q/k/v to fp16 for SageAttention

检查：

- 两个都关：出图与未装这个扩展时一致。
- 只开计算 fp16：日志有 `Diffusion compute dtype ... -> float16`。连续两次出图，第二次不重新加载模型。关掉后，下一次出图会重新加载。
- 只开 Sage cast，以及两个都开：采样不报 Half/Float。
- 低显存先出一张，再开计算 fp16，再出一张：不报 Half/Float。
- Hires fix 开启后，第二遍的计算 dtype 仍是 fp16。
- refiner 换成另一份模型后，计算 dtype 仍是 fp16。
- ControlNet 与 fp16 UNet 一起采样，不报 dtype 错误。
- Nunchaku 模型：日志写明跳过 fp16 compute，出图不崩。
- 注意力后端不是 SageAttention：有警告，q/k/v 不转换，出图仍成功。
- API 请求用 `override_settings` 只对这一次打开开关。请求结束、UI 选项恢复之后，模型保持已打过的补丁。下一次生成看到开关关闭时才重新加载。
- 在扩展列表里禁用本扩展并重启：设置项消失，行为回到官方 Forge Neo。
