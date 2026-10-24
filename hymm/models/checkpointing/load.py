"""Load the public DiT and text encoder from ordinary model assets."""

from __future__ import annotations

import re
from pathlib import Path

from safetensors.torch import load_file

from configs.constants import PROMPT_TEMPLATE
from hymm.models.dit.modules import load_model
from hymm.models.text_encoder import TextEncoder
from hymm.utils.camera_utils import is_camera_param


def _weight_path(dit_weight: str) -> Path:
    path = Path(dit_weight).expanduser()
    if path.is_dir():
        path = path / "diffusion_pytorch_model.safetensors"
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors DiT checkpoint: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"DiT checkpoint not found: {path}")
    return path


def load_hunyuan_state_dict(
    args,
    model,
    dit_weight,
    convert_state_dict_in_out=False,
    convert_state_dict_16168_to_16164=False,
    logger=None,
    cast_to_model_dtype=False,
):
    """Load a compatible safetensors checkpoint into the DiT.

    A base DiT may omit newly initialized camera modules. Other missing model
    parameters and tensor-shape mismatches indicate incompatible weights.
    ``cast_to_model_dtype`` streams the file tensor by tensor and casts to the
    model's parameter dtype on the way in, halving host RAM for FP32 files.
    """
    if convert_state_dict_in_out or convert_state_dict_16168_to_16164:
        raise ValueError(
            "Public checkpoints must match the model architecture; "
            "legacy weight-shape conversion is unsupported"
        )
    path = _weight_path(dit_weight)
    expected = model.state_dict()

    def _normalize(key):
        match = camera_pattern.match(key)
        if match:
            key = f"action_modules.{match.group(1)}.{match.group(2)}"
        key = key.replace("._checkpoint_wrapped_module.", ".")
        return key.replace("_checkpoint_wrapped_module.", "")

    camera_pattern = re.compile(r"^double_blocks\.(\d+)\.action_module\.(.+)$")
    remapped = {}
    if cast_to_model_dtype:
        # Read tensor by tensor and cast each floating tensor to the model's
        # dtype as it arrives, so an FP32 checkpoint costs one BF16 copy of host
        # RAM instead of the full FP32 dict plus the model (34 GB + 17 GB for
        # the 8.5B DiT). The values are identical to casting after load.
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                key = _normalize(raw_key)
                if key in remapped:
                    raise ValueError(f"Duplicate DiT key after normalization: {key}")
                tensor = handle.get_tensor(raw_key)
                target = expected.get(key)
                if target is not None and tensor.is_floating_point() and tensor.dtype != target.dtype:
                    tensor = tensor.to(dtype=target.dtype)
                remapped[key] = tensor
    else:
        weights = load_file(str(path))
        for raw_key, tensor in weights.items():
            key = _normalize(raw_key)
            if key in remapped:
                raise ValueError(f"Duplicate DiT key after normalization: {key}")
            remapped[key] = tensor
    if not remapped:
        raise ValueError(f"DiT checkpoint contains no tensors: {path}")

    unexpected = sorted(set(remapped) - set(expected))
    for key in unexpected:
        del remapped[key]
    mismatched = [
        key for key, tensor in remapped.items() if tensor.shape != expected[key].shape
    ]
    if mismatched:
        detail = ", ".join(
            f"{key}: {tuple(remapped[key].shape)} vs {tuple(expected[key].shape)}"
            for key in mismatched[:5]
        )
        raise ValueError(f"Incompatible DiT tensor shapes in {path}: {detail}")

    missing = sorted(set(expected) - set(remapped))
    parameter_names = set(dict(model.named_parameters()))
    missing_required = [
        key for key in missing if key in parameter_names and not is_camera_param(key)
    ]
    if missing_required:
        raise ValueError(
            f"Missing {len(missing_required)} non-camera DiT parameters in {path}: "
            f"{missing_required[:10]}"
        )

    model.load_state_dict(remapped, strict=False)
    if logger is not None:
        logger.info(
            f"Loaded {len(remapped)} DiT tensors from {path}; "
            f"new camera tensors={sum(is_camera_param(k) for k in missing)}, "
            f"unexpected keys ignored={len(unexpected)}"
        )
    return model


def load_transformer(
    model_type,
    dit_weight,
    factor_kwargs=None,
    convert_state_dict_in_out=False,
    convert_state_dict_16168_to_16164=False,
    args=None,
    logger=None,
    cast_to_model_dtype=False,
):
    """Construct the public DiT and load safetensors weights."""
    transformer = load_model(model_type, args, factor_kwargs)
    return load_hunyuan_state_dict(
        args,
        transformer,
        dit_weight,
        convert_state_dict_in_out=convert_state_dict_in_out,
        convert_state_dict_16168_to_16164=convert_state_dict_16168_to_16164,
        logger=logger,
        cast_to_model_dtype=cast_to_model_dtype,
    )


def load_text_encoder(device, logger, args):

    # Use the original TextEncoder for non-distributed mode
    text_encoder = TextEncoder(
        text_encoder_type=args.text_encoder_type,
        text_encoder_path=args.text_encoder.model_name,
        max_length=args.text_encoder.text_len
        if not args.text_encoder.use_ll_cap_dataset
        else args.text_encoder.text_len_ll_cap,
        text_encoder_precision=args.text_encoder.text_encoder_precision,
        tokenizer_type=args.text_encoder.tokenizer,
        prompt_template=PROMPT_TEMPLATE[args.text_encoder.prompt_template]
        if args.text_encoder.prompt_template
        else None,
        prompt_template_video=PROMPT_TEMPLATE[args.text_encoder.prompt_template_video]
        if args.text_encoder.prompt_template_video
        else None,
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        logger=logger,
        device=device,
    )

    return text_encoder


def get_no_split_modules(transformer):
    from hymm.models.dit.nomas_cpt import (
        CPTHYNomasDiffusionTransformer,
        MMDoubleStreamBlock,
        MMSingleStreamBlock,
    )

    if isinstance(transformer, CPTHYNomasDiffusionTransformer):
        return (MMDoubleStreamBlock, MMSingleStreamBlock)
    raise ValueError(f"Unsupported transformer type: {type(transformer)}")
