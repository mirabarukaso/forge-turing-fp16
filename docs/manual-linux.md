# Linux：不使用脚本的手动步骤

在 Forge Neo 根目录执行。把 `/path/to/forge-neo` 换成你的路径。`<` 在 bash 里也是重定向，包装名必须加引号。

## 1. 虚拟环境

```bash
python3 -m venv venv
PY=/path/to/forge-neo/venv/bin/python
```

## 2. PyTorch

已经安装就不要重装。缺失时使用 Forge Neo 的 `TORCH_COMMAND`。没有设置这个环境变量时，与当前默认一致：

```bash
"$PY" -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
```

## 3. 确认是 Turing

```bash
"$PY" -c "import torch; print(torch.cuda.get_device_capability(0))"
```

输出必须是 `(7, 5)`。不是的话停在这里。

Turing 不是第一块可见 GPU 时，先 `export CUDA_VISIBLE_DEVICES=1`，再探测和安装。

## 4. Triton

错误写法会让 bash 去读一个叫 `3.3` 的文件：

```bash
"$PY" -m pip install -U --no-deps triton<3.3
```

正确写法：

```bash
"$PY" -m pip install -U --no-deps "triton<3.3"
```

Linux 包装名是 `triton`，不是 `triton-windows`。已经低于 3.3 时跳过。

uv：

```bash
uv pip install --python "$PY" -U --no-deps "triton<3.3"
```

## 5. SageAttention

Linux 没有 Windows 那种 post3 wheel。安装 PyPI 上的 2.2.0：

```bash
"$PY" -m pip install -U --no-deps "sageattention==2.2.0"
```

`pip show sageattention` 已经是 2.2.0 时跳过。

## 6. xformers

只在缺失时安装，索引与 torch 的 CUDA 一致：

```bash
"$PY" -m pip install xformers==0.0.35 --extra-index-url https://download.pytorch.org/whl/cu130
```

## 7. Sage 导入 stub

复制 `installer/turing_sage_stub.py` 到 venv 的 site-packages。路径类似：

```text
venv/lib/python3.12/site-packages/turing_sage_stub.py
```

Python 小版本以你的 venv 为准。同一目录新建 `turing_sage_stub.pth`：

```text
import turing_sage_stub
```

钩子在 SageAttention 加载 `_fused` 时才检查 sm75。失败则换成空模块，让 Triton kernel 可以导入。其他 GPU 仍使用真实扩展。

## 8. 启动参数

`webui-user.sh`：

```bash
export COMMANDLINE_ARGS="--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install"
```

`--skip-install` 阻止下次启动把 Triton 升到 3.3 以上，或把 Sage 换成 Forge 默认的新包装。`requirements.txt` 和扩展 `install.py` 也不会自动跑。需要时：

```bash
"$PY" -m pip install -r requirements.txt
```

不要加 `-U`。

## 9. 扩展

仓库放在 `extensions/forge-turing-fp16`，启动 `webui.sh`。设置页 Optimizations 里会出现两个开关。
