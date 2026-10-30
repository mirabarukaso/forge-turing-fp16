# Windows：不使用脚本的手动步骤

下面的命令都在 Forge Neo 根目录执行。把 `F:\sd-webui-forge-neo` 换成你的路径。Python 调用使用参数形式，不要把带 `<` 的包装名拼进 `cmd /c` 的一整行字符串。

## 1. 虚拟环境

如果还没有 `venv\Scripts\python.exe`：

```bat
python -m venv venv
```

后文用：

```bat
set PY=F:\sd-webui-forge-neo\venv\Scripts\python.exe
```

## 2. PyTorch

已经安装就不要重装。缺失时，与当前 Forge Neo 默认一致：

```bat
"%PY%" -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
```

环境变量 `TORCH_COMMAND` 有值时，用那个命令代替上面这一行。

## 3. 确认是 Turing

```bat
"%PY%" -c "import torch; print(torch.cuda.get_device_capability(0))"
```

输出必须是 `(7, 5)`。不是的话停在这里，不要改 Sage 和 Triton。

Turing 卡不是当前可见的第一块 GPU 时，先在同一个终端里执行 `set CUDA_VISIBLE_DEVICES=1`（换成实际编号），再做上面的探测和后面的安装。

## 4. Triton

在 **cmd.exe** 里，下面这一行是错的：

```bat
"%PY%" -m pip install -U --no-deps triton-windows<3.3
```

`<3.3` 会被当成“从文件 `3.3` 读入标准输入”，报 `The system cannot find the file specified.`

用引号：

```bat
"%PY%" -m pip install -U --no-deps "triton-windows<3.3"
```

已经安装且版本低于 3.3 时，跳过这一步。

用 uv 时同样要带引号：

```bat
uv pip install --python "%PY%" -U --no-deps "triton-windows<3.3"
```

## 5. SageAttention 2.2 post3

按 torch 的 CUDA 选一个 wheel。`+cu130` 用最后一条。

```text
cu124  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu124torch2.5.1.post3-cp39-abi3-win_amd64.whl
cu126  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl
cu128  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl
cu130  https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu130torch2.9.0.post3-cp39-abi3-win_amd64.whl
```

```bat
"%PY%" -m pip install -U --no-deps https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu130torch2.9.0.post3-cp39-abi3-win_amd64.whl
```

`pip show sageattention` 的版本里已经有 `post3` 时，跳过。

## 6. xformers

只在缺失时安装。索引跟上一步 torch 的 CUDA：

```bat
"%PY%" -m pip install xformers==0.0.35 --extra-index-url https://download.pytorch.org/whl/cu130
```

## 7. Sage 导入 stub

把本仓库的 `installer\turing_sage_stub.py` 复制到：

```text
venv\Lib\site-packages\turing_sage_stub.py
```

在同一目录新建 `turing_sage_stub.pth`，内容只有一行：

```text
import turing_sage_stub
```

这个文件在解释器启动时挂上导入钩子。它不在启动瞬间导入 torch。只有 SageAttention 去加载 `_fused` 时才检查是不是 sm75；加载失败才换成空模块，让 fp16 Triton kernel 继续导入。其他 GPU 仍使用真实扩展。

## 8. 启动参数

`webui-user.bat`：

```bat
set COMMANDLINE_ARGS=--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install
```

`--skip-install` 是为了阻止下次启动把 Sage 升到 post6、把 Triton 升到 3.3 以上。副作用是 `requirements.txt` 和扩展 `install.py` 也不再自动执行。需要时手动：

```bat
"%PY%" -m pip install -r requirements.txt
```

不要加 `-U`。`requirements.txt` 里没有 sage / triton。

## 9. 扩展

把本仓库放到 `extensions\forge-turing-fp16`，启动 `webui.bat`。设置页 Optimizations 里会出现两个开关。
