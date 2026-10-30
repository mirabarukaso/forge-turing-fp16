# Linux：用脚本安装

在 Forge Neo 根目录已经能创建 `venv` 的前提下使用。脚本用参数数组调用 pip 或 uv，`triton<3.3` 不会被 bash 当成重定向。

## 1. 准备扩展

```bash
cd /path/to/forge-neo/extensions
git clone https://github.com/mirabarukaso/forge-turing-fp16.git
chmod +x forge-turing-fp16/installer/install-turing.sh
```

## 2. 关掉 WebUI 的自动安装

编辑 `webui-user.sh`。Turing 上建议：

```bash
export COMMANDLINE_ARGS="--api --sage --xformers --uv --pin-shared-memory --cuda-malloc --skip-install"
```

不要加 `--reinstall-torch`。`--skip-install` 会跳过 Sage、Triton、xformers、`requirements.txt` 和扩展的 `install.py`。torch 缺失，或带了 `--reinstall-torch` 时，Forge 仍会自己装 torch。

升级 Forge Neo 之后，需要依赖更新时给脚本加 `--requirements`。

## 3. 运行脚本

```bash
./installer/install-turing.sh --forge-root /path/to/forge-neo --uv
```

扩展已经放在 `extensions/forge-turing-fp16` 时可以省略 `--forge-root`。没有 uv 时去掉 `--uv`。

可选参数：

- `--device-id 1`：探测前设置 `CUDA_VISIBLE_DEVICES`。
- `--requirements`：执行 `pip install -r requirements.txt`，不加 `-U`。

算力不是 7.5 时脚本退出，并且不改 Sage、Triton、xformers 和导入 stub。Linux 上 Sage 装的是 PyPI 的 `sageattention==2.2.0`，Triton 约束是 `triton<3.3`，不是 `triton-windows`。

## 4. 确认

```bash
/path/to/forge-neo/venv/bin/python -m pip show sageattention triton xformers
```

期望：

- `sageattention` 是 2.2.0
- `triton` 小于 3.3
- `venv/lib/pythonX.Y/site-packages/turing_sage_stub.pth` 存在

启动 `webui.sh`。CUDA 扩展加载失败、改走 Triton 时，日志里有：

```text
SageAttention CUDA extension unavailable; using Triton kernels
```
