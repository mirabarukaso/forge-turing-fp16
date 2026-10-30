"""Sticky fp16 compute and SageAttention q/k/v cast for a loaded Forge Neo UNet.

The switches stay on the live model after override_settings restores the UI
options. The next generation that sees a switch off reloads the checkpoint
instead of converting the weights back.

Upstream hooks this file wraps, re-check after each Forge Neo update:
- modules.sd_models.forge_model_reload
- modules.processing.forge_model_reload (imported by name, not looked up later)
- backend.modules.k_model.KModel.apply_model
- backend.attention.attention_sage and attention.attention_function
"""

from __future__ import annotations

import contextvars
import logging

import torch

sage_fp16_cast_active: contextvars.ContextVar[bool] = contextvars.ContextVar("sage_fp16_cast_active", default=False)

logger = logging.getLogger("forge_turing_fp16")


def _setup_logger() -> None:
    try:
        from backend.logging import setup_logger
    except Exception:
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return
    setup_logger(logger)


def _dtype_name(dt) -> str:
    return str(dt).replace("torch.", "")


def _is_nunchaku(module: torch.nn.Module) -> bool:
    return type(module).__module__ == "backend.nn.svdq"


def cast_floating(obj, dtype: torch.dtype):
    """Cast floating tensors inside control residuals to the UNet compute dtype."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype != dtype:
            return obj.to(dtype=dtype)
        return obj
    if isinstance(obj, dict):
        return {key: cast_floating(value, dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [cast_floating(value, dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(cast_floating(value, dtype) for value in obj)
    return obj


def _enable_manual_cast(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Embedding):
            continue
        if hasattr(layer, "parameters_manual_cast") and not layer.parameters_manual_cast:
            layer.parameters_manual_cast = True
        # An offloaded layer can already be True while prev_parameters_manual_cast
        # is still False. wipe_lowvram_weight() would restore False and the next
        # fp16 matmul fails. Keep the saved value on as well.
        if hasattr(layer, "prev_parameters_manual_cast"):
            layer.prev_parameters_manual_cast = True


def apply_fp16_switch(sd_model, desired: tuple[bool, bool], already: tuple[bool, bool] = (False, False)) -> bool:
    """Enable compute fp16 and/or the Sage fp16 cast. Returns True if the module changed."""
    want_compute, want_sage = desired
    had_compute, had_sage = already
    changed = False
    kmodel = sd_model.forge_objects.unet.model
    diffusion = kmodel.diffusion_model

    if want_compute and not had_compute:
        if _is_nunchaku(diffusion):
            logger.info("Nunchaku UNet keeps its compute dtype; fp16 compute switch skipped")
        elif kmodel.computation_dtype != torch.float16:
            old = kmodel.computation_dtype
            if getattr(diffusion, "storage_dtype", None) != torch.float16:
                _enable_manual_cast(diffusion)
            kmodel.computation_dtype = torch.float16
            diffusion.computation_dtype = torch.float16
            logger.info("Diffusion compute dtype %s -> float16", _dtype_name(old))
            changed = True

    if want_sage and not had_sage and not getattr(kmodel, "sage_fp16_cast", False):
        kmodel.sage_fp16_cast = True
        from backend import attention

        if attention.attention_function is attention.attention_sage:
            logger.info("SageAttention fp16 cast enabled for this UNet")
        else:
            logger.warning("SageAttention fp16 cast is on, but the active attention backend is not SageAttention; q/k/v will not be cast")
        changed = True

    return changed


def _desired() -> tuple[bool, bool]:
    from modules import shared

    return bool(getattr(shared.opts, "forge_compute_fp16", False)), bool(getattr(shared.opts, "forge_sage_fp16_cast", False))


def reconcile_fp16_switch() -> None:
    """Match the loaded UNet to the fp16 switches.

    Turning a switch on patches the current model once. Turning it off does not
    undo that patch here; a later generation whose options are off reloads.
    """
    from modules import sd_models
    from backend import memory_management

    sd_model = sd_models.model_data.sd_model
    if sd_model is None or not getattr(sd_model, "forge_objects", None):
        return

    desired = _desired()
    applied = getattr(sd_model, "forge_fp16_switch", (False, False))
    if desired == applied:
        return

    turning_off = (applied[0] and not desired[0]) or (applied[1] and not desired[1])
    if turning_off and getattr(sd_model, "forge_fp16_switch_dirty", False):
        memory_management.logger.info("FP16 switch is off; reloading the diffusion model")
        sd_models.model_data.forge_hash = ""
        sd_model, _ = _call_original_reload()
        if desired == (False, False):
            return
        sd_model.forge_fp16_switch_dirty = apply_fp16_switch(sd_model, desired) or getattr(sd_model, "forge_fp16_switch_dirty", False)
        sd_model.forge_fp16_switch = desired
        return

    if turning_off:
        sd_model.forge_fp16_switch = desired
        return

    sd_model.forge_fp16_switch_dirty = apply_fp16_switch(sd_model, desired, already=applied) or getattr(sd_model, "forge_fp16_switch_dirty", False)
    sd_model.forge_fp16_switch = desired


def _call_original_reload():
    from modules import sd_models

    original = getattr(sd_models.forge_model_reload, "_forge_turing_fp16_original", None)
    if original is None:
        raise RuntimeError("forge_model_reload is not wrapped by forge-turing-fp16")
    return original()


def _wrap_reload(original):
    def forge_model_reload():
        model, changed = original()
        reconcile_fp16_switch()
        return model, changed

    forge_model_reload._forge_turing_fp16 = True
    forge_model_reload._forge_turing_fp16_original = original
    return forge_model_reload


def _wrap_apply_model(original):
    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        if control is not None and getattr(self, "computation_dtype", None) == torch.float16:
            control = cast_floating(control, torch.float16)
        token = sage_fp16_cast_active.set(True) if getattr(self, "sage_fp16_cast", False) else None
        try:
            return original(self, x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)
        finally:
            if token is not None:
                sage_fp16_cast_active.reset(token)

    apply_model._forge_turing_fp16 = True
    return apply_model


def _wrap_attention_sage(original):
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype
        cast = sage_fp16_cast_active.get() and in_dtype != torch.float16
        if cast:
            q = q.to(dtype=torch.float16)
            k = k.to(dtype=torch.float16)
            v = v.to(dtype=torch.float16)
        out = original(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        if cast and isinstance(out, torch.Tensor) and out.dtype != in_dtype:
            out = out.to(dtype=in_dtype)
        return out

    attention_sage._forge_turing_fp16 = True
    return attention_sage


def install_patches() -> None:
    """Wrap Forge Neo entry points. Safe to call again after an extension reload."""
    from backend import attention
    from backend.modules.k_model import KModel
    from modules import processing, sd_models

    _setup_logger()

    if not getattr(sd_models.forge_model_reload, "_forge_turing_fp16", False):
        wrapped_reload = _wrap_reload(sd_models.forge_model_reload)
        sd_models.forge_model_reload = wrapped_reload
        processing.forge_model_reload = wrapped_reload

    if not getattr(KModel.apply_model, "_forge_turing_fp16", False):
        KModel.apply_model = _wrap_apply_model(KModel.apply_model)

    if not getattr(attention.attention_sage, "_forge_turing_fp16", False):
        wrapped_sage = _wrap_attention_sage(attention.attention_sage)
        if attention.attention_function is attention.attention_sage:
            attention.attention_function = wrapped_sage
        attention.attention_sage = wrapped_sage

    logger.info("Forge Turing FP16 patches installed")
