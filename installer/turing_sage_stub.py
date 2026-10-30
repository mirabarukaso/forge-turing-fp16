"""Let SageAttention import on Turing when its CUDA extension cannot load.

A .pth file imports this module at interpreter startup. The finder does not
import torch until SageAttention actually loads _fused, which is after Forge
has applied CUDA malloc and CUDA_VISIBLE_DEVICES.

On sm75, a failed _fused load is replaced with an empty module so the fp16
Triton kernels can import. Other GPUs keep the real extension, including its
errors.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys

_INSTALLED = False
_SM75 = None


def _is_sm75() -> bool:
    global _SM75
    if _SM75 is not None:
        return _SM75
    try:
        import torch
    except Exception:
        _SM75 = False
        return False
    try:
        cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        _SM75 = bool(cuda and torch.cuda.get_device_capability(0) == (7, 5))
    except Exception:
        _SM75 = False
    return _SM75


def _log_fallback() -> None:
    message = "SageAttention CUDA extension unavailable; using Triton kernels"
    print(message)
    try:
        import logging

        logging.getLogger("forge_turing_fp16").info(message)
    except Exception:
        pass


class _EmptyLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        _log_fallback()


class _ChainedLoader(importlib.abc.Loader):
    def __init__(self, real):
        self.real = real
        self.failed = False

    def create_module(self, spec):
        create = getattr(self.real, "create_module", None)
        if create is None:
            return None
        try:
            return create(spec)
        except Exception:
            if not _is_sm75():
                raise
            self.failed = True
            return None

    def exec_module(self, module):
        if self.failed:
            _log_fallback()
            return
        try:
            self.real.exec_module(module)
        except Exception:
            if not _is_sm75():
                raise
            module.__dict__.clear()
            _log_fallback()


class _FusedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "sageattention._fused":
            return None

        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            spec = None
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is not None and spec.loader is not None:
            spec.loader = _ChainedLoader(spec.loader)
            return spec
        if _is_sm75():
            return importlib.machinery.ModuleSpec(fullname, _EmptyLoader())
        return None


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    sys.meta_path.insert(0, _FusedFinder())
    _INSTALLED = True


install()
