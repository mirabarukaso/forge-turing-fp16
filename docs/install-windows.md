# Windows：用脚本安装

在 Forge Neo 根目录已经能创建 `venv` 的前提下使用。脚本不会走 `cmd.exe`，版本约束里的 `<` 不会被当成重定向。

## 1. 准备扩展

把本仓库放到：

```text
<Forge Neo>\extensions\forge-turing-fp16
```

## 2. 关掉 WebUI 的自动安装

编辑 `webui-user.bat`，保留你原来的参数，并加上 `--skip-install`。这台调试机器上的一行是：

```bat
set COMMANDLINE_ARGS=--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install
```

不要加 `--reinstall-torch`。`--skip-install` 会跳过 Sage、Triton、xformers、`requirements.txt` 和扩展的 `install.py`。它挡不住 PyTorch：torch 缺失，或带了 `--reinstall-torch` 时，Forge 仍会自己装 torch。

升级 Forge Neo 之后，`requirements.txt` 和扩展 `install.py` 也不会自动跑。需要时给脚本加 `-Requirements`。

## 3. 运行脚本

在 PowerShell 里：

```powershell
cd W:\Github\forge-turing-fp16
.\installer\install-turing.ps1 -ForgeRoot F:\sd-webui-forge-neo -UseUv
```

扩展已经放在 `extensions\forge-turing-fp16` 时可以省略 `-ForgeRoot`。当前 bat 使用 `--uv`，所以加上 `-UseUv`。没有 uv 时去掉这个开关，脚本会改用 venv 里的 pip。

可选参数：

- `-DeviceId 1`：探测前设置 `CUDA_VISIBLE_DEVICES`。Turing 不是系统里的第一块可见 GPU 时才需要。
- `-Requirements`：再执行一次 `pip install -r requirements.txt`，不加 `-U`，不会把已经固定的 Sage / Triton 升回去。

算力不是 7.5 时脚本退出，并且不改 Sage、Triton、xformers 和导入 stub。torch 本来就缺失时，脚本会先装 Forge Neo 当前默认的 torch，才能做这次探测。

## 4. 确认

```powershell
F:\sd-webui-forge-neo\venv\Scripts\python.exe -m pip show sageattention triton-windows xformers
```

期望：

- `sageattention` 版本字符串里有 `post3`
- `triton-windows` 小于 3.3
- `venv\Lib\site-packages\turing_sage_stub.pth` 存在

然后启动 `webui.bat`。Sage 导入走 Triton 时，日志里有：

```text
SageAttention CUDA extension unavailable; using Triton kernels
```

CUDA 扩展本身能加载时不会出现这行，这是正常的。
