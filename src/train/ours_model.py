"""Load Qwen2.5-Omni / Phi-4-MM / Qwen2-Audio, freeze towers, attach LLM LoRA.

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LLM_SELF_ATTN_PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
FORBIDDEN_LORA_SUBSTR = ("audio_tower", "visual", "talker", "audio_encoder")
QWEN2_AUDIO_FORBIDDEN_LORA_SUBSTR = FORBIDDEN_LORA_SUBSTR + (
    "multi_modal_projector",
)
PHI4_FORBIDDEN_LORA_SUBSTR = FORBIDDEN_LORA_SUBSTR + (
    "embed_tokens_extend",
    "vision_embed",
    "audio_embed",
    "conformer",
    "vision_model",
    "audio_model",
    "img_projection",
    "audio_projection",
)
PHI4_LLM_SELF_ATTN_RE = re.compile(
    r"layers\.\d+\.self_attn\.(qkv_proj|o_proj)$"
)
BACKBONE_QWEN25_OMNI = "qwen25_omni"
BACKBONE_PHI4_MM = "phi4_mm"
BACKBONE_QWEN2_AUDIO = "qwen2_audio"


def load_omni_processor(model_dir: Path) -> Any:
    """Load the Omni processor from a local directory."""

    from transformers import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor.from_pretrained(str(model_dir), trust_remote_code=True)


def load_omni_thinker(model_dir: Path, *, dtype: torch.dtype) -> Any:
    """Load Thinker (text generation path) in bf16/fp16 on CUDA."""

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def freeze_non_lora_modules(model: Any) -> None:
    """Freeze all parameters before LoRA injection (encoder/adaptor stay frozen)."""

    for parameter in model.parameters():
        parameter.requires_grad = False


def collect_phi4_llm_self_attn_proj_names(model: Any) -> list[str]:
    """Return Phi-4 LLM self-attn names (qkv_proj / o_proj, not encoders)."""

    names: list[str] = []
    qkv_names: list[str] = []
    for name, _module in model.named_modules():
        if name.endswith("qkv_proj"):
            qkv_names.append(name)
        if any(part in name for part in PHI4_FORBIDDEN_LORA_SUBSTR):
            continue
        if PHI4_LLM_SELF_ATTN_RE.search(name):
            names.append(name)
    if not names:
        raise RuntimeError(
            "No Phi-4 LLM self-attn LoRA targets matched layers.*.self_attn."
            f" Example qkv_proj names: {qkv_names[:12]}"
        )
    return names


def _forbidden_lora_substr(backbone: str) -> tuple[str, ...]:
    """Return substrings that must never appear in trainable LoRA names."""

    if backbone == BACKBONE_PHI4_MM:
        return PHI4_FORBIDDEN_LORA_SUBSTR
    if backbone == BACKBONE_QWEN2_AUDIO:
        return QWEN2_AUDIO_FORBIDDEN_LORA_SUBSTR
    return FORBIDDEN_LORA_SUBSTR


def collect_llm_self_attn_proj_names(
    model: Any,
    *,
    backbone: str = BACKBONE_QWEN25_OMNI,
) -> list[str]:
    """Return LLM self-attn projection module names (not audio/visual towers)."""

    if backbone == BACKBONE_PHI4_MM:
        return collect_phi4_llm_self_attn_proj_names(model)

    forbidden = _forbidden_lora_substr(backbone)
    strict: list[str] = []
    fallback: list[str] = []
    fallback_re = re.compile(
        r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
    )
    q_proj_names: list[str] = []
    for name, _module in model.named_modules():
        if name.endswith("q_proj"):
            q_proj_names.append(name)
        if any(part in name for part in forbidden):
            continue
        if LLM_SELF_ATTN_PROJ_RE.fullmatch(name):
            strict.append(name)
        elif fallback_re.search(name):
            fallback.append(name)
    # Qwen2-Audio LLM is under language_model.*; Omni strict regex misses it.
    names = fallback if backbone == BACKBONE_QWEN2_AUDIO else (strict or fallback)
    if not names:
        raise RuntimeError(
            "No LLM self-attn LoRA targets matched model.layers.*.self_attn."
            f" Example q_proj names: {q_proj_names[:12]}"
        )
    return names


def assert_trainable_lora_is_llm_attention(
    model: Any,
    *,
    backbone: str = BACKBONE_QWEN25_OMNI,
) -> list[str]:
    """Abort if any trainable parameter sits in audio/visual/talker modules."""

    forbidden = _forbidden_lora_substr(backbone)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    leaked = [
        name
        for name in trainable
        if any(part in name for part in forbidden)
    ]
    if leaked:
        raise RuntimeError(
            "LoRA leaked into non-LLM modules. "
            f"Examples: {leaked[:8]}"
        )
    if not trainable:
        raise RuntimeError("LoRA attached but no parameter requires grad.")
    return trainable


def attach_attention_lora(
    model: Any,
    *,
    backbone: str = BACKBONE_QWEN25_OMNI,
) -> Any:
    """Attach LoRA adapters on LLM Attention projections only."""

    from peft import LoraConfig, get_peft_model

    target_modules = collect_llm_self_attn_proj_names(model, backbone=backbone)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    trainable = assert_trainable_lora_is_llm_attention(
        peft_model, backbone=backbone
    )
    peft_model.print_trainable_parameters()
    print(
        f"[ours] LoRA targets={len(target_modules)} "
        f"trainable_tensors={len(trainable)} "
        f"first={trainable[0]}",
        flush=True,
    )
    return peft_model


def _noop_adapter_switch(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder for Phi-4 official speech/vision LoRA switching."""

    return None


def disable_phi4_official_lora_switch(model: Any) -> None:
    """Stop Phi-4 forward() from retargeting LoRA after speech merge.

    Official ``forward`` calls ``set_lora_adapter('speech')`` or
    ``unset_lora_adapter()`` from ``input_mode``. After speech-LoRA is
    merged, those calls would disable or miss the Ours adapter.
    """

    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(root, "set_lora_adapter"):
        root.set_lora_adapter = _noop_adapter_switch
    if hasattr(root, "unset_lora_adapter"):
        root.unset_lora_adapter = _noop_adapter_switch


def load_phi4_processor(model_dir: Path) -> Any:
    """Load the Phi-4-MM processor from a local directory."""

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)


def patch_dynamic_cache_get_usable_length() -> None:
    """Phi-4 local modeling still calls DynamicCache.get_usable_length.

    otherwise fails on every row with DynamicCache.
    """

    from transformers.cache_utils import Cache, DynamicCache

    def get_usable_length(self, kv_length: int, layer_idx: int = 0) -> int:
        seq = 0
        if hasattr(self, "get_seq_length"):
            try:
                seq = int(self.get_seq_length(layer_idx))
            except TypeError:
                seq = int(self.get_seq_length())
        return int(kv_length) + seq

    DynamicCache.get_usable_length = get_usable_length
    Cache.get_usable_length = get_usable_length


def patch_phi4mm_model_for_peft20() -> None:
    """Let peft 0.20 wrap official Phi4MMModel vision/speech adapters.

    model. Official Phi4MMModel does not define it. Do not scan all
    sys.modules: a torch custom class is also named Phi4MMModel.
    """

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    import peft.peft_model as peft_model

    if getattr(peft_model.PeftModelForCausalLM, "_realmg_phi4_patched", False):
        return
    original_init = peft_model.PeftModelForCausalLM.__init__

    def patched_init(self, model, *args: Any, **kwargs: Any) -> None:
        cls = type(model)
        if "prepare_inputs_for_generation" not in cls.__dict__:
            cls.prepare_inputs_for_generation = prepare_inputs_for_generation
        original_init(self, model, *args, **kwargs)

    peft_model.PeftModelForCausalLM.__init__ = patched_init
    peft_model.PeftModelForCausalLM._realmg_phi4_patched = True


def patch_phi4_logits_to_keep(model: Any) -> None:
    """Coerce None num_logits_to_keep so Phi-4 forward can slice hidden states.

    Official modeling does ``hidden_states[:, -num_logits_to_keep:, :]``.
    transformers 4.57 generate may pass None / logits_to_keep instead.
    """

    import types

    original_forward = model.__class__.forward

    def wrapped_forward(self, *args: Any, **kwargs: Any) -> Any:
        keep = kwargs.get("num_logits_to_keep")
        if keep is None:
            keep = kwargs.get("logits_to_keep")
        if keep is None:
            keep = 0
        kwargs["num_logits_to_keep"] = keep
        return original_forward(self, *args, **kwargs)

    model.forward = types.MethodType(wrapped_forward, model)


def merge_phi4_speech_lora(model: Any, model_dir: Path) -> Any:
    """Load official speech-LoRA weights and merge them into dense LLM weights.

    Official ``Phi4MMForCausalLM.__init__`` calls ``get_peft_model`` but never
    assigns the wrapper back to ``self.model``. After ``from_pretrained``,
    ``model.model`` is still ``Phi4MMModel``. Attach ``speech-lora`` on the
    causal LM and merge so Teacher-text matches the speech-tuned frozen model.
    """

    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file

    speech_dir = Path(model_dir) / "speech-lora"
    adapter_file = speech_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"Missing Phi-4 speech-lora weights at {adapter_file}"
        )
    inner = model.model
    print(
        f"[ours] phi4 inner={type(inner).__name__} "
        f"is_peft={isinstance(inner, PeftModel)}",
        flush=True,
    )
    if isinstance(inner, PeftModel):
        inner.set_adapter("speech")
        set_peft_model_state_dict(
            inner, load_file(str(adapter_file)), adapter_name="speech"
        )
        model.model = inner.merge_and_unload()
        patch_phi4_logits_to_keep(model)
        return model
    wrapped = PeftModel.from_pretrained(
        model, str(speech_dir), adapter_name="speech"
    )
    merged = wrapped.merge_and_unload()
    patch_phi4_logits_to_keep(merged)
    return merged


def load_phi4_with_speech_merged(model_dir: Path, *, dtype: torch.dtype) -> Any:
    """Load Phi-4-MM and merge official speech-LoRA into dense weights.

    LoRA. Do not load this on the 2GiB CPU cgroup.
    """

    from transformers import AutoModelForCausalLM

    patch_dynamic_cache_get_usable_length()
    patch_phi4mm_model_for_peft20()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        _attn_implementation="sdpa",
    )
    return merge_phi4_speech_lora(model, Path(model_dir))


def build_phi4_student(model_dir: Path, *, device: torch.device) -> tuple[Any, Any]:
    """Return (processor, peft student) for Phi-4-MM Ours training."""

    if device.type != "cuda":
        raise RuntimeError("Phi-4-MM Ours loading requires CUDA.")
    dtype = torch.bfloat16
    processor = load_phi4_processor(model_dir)
    model = load_phi4_with_speech_merged(model_dir, dtype=dtype)
    freeze_non_lora_modules(model)
    student = attach_attention_lora(model, backbone=BACKBONE_PHI4_MM)
    disable_phi4_official_lora_switch(student)
    student.to(device)
    student.train()
    return processor, student


def load_qwen2_audio_processor(model_dir: Path) -> Any:
    """Load the Qwen2-Audio-Instruct processor from a local directory."""

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_dir))


def load_qwen2_audio_model(model_dir: Path, *, dtype: torch.dtype) -> Any:
    """Load Qwen2AudioForConditionalGeneration in bf16/fp16 on CUDA."""

    from transformers import Qwen2AudioForConditionalGeneration

    return Qwen2AudioForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
    )


def build_qwen2_audio_student(
    model_dir: Path, *, device: torch.device
) -> tuple[Any, Any]:
    """Return (processor, peft student) for Qwen2-Audio Ours training."""

    if device.type != "cuda":
        raise RuntimeError("Qwen2-Audio Ours loading requires CUDA.")
    dtype = torch.bfloat16
    processor = load_qwen2_audio_processor(model_dir)
    model = load_qwen2_audio_model(model_dir, dtype=dtype)
    freeze_non_lora_modules(model)
    student = attach_attention_lora(model, backbone=BACKBONE_QWEN2_AUDIO)
    student.to(device)
    student.train()
    return processor, student


def build_ours_student(
    model_dir: Path,
    *,
    device: torch.device,
    backbone: str = BACKBONE_QWEN25_OMNI,
) -> tuple[Any, Any]:
    """Return (processor, peft student) ready for Ours training."""

    if backbone == BACKBONE_PHI4_MM:
        return build_phi4_student(model_dir, device=device)
    if backbone == BACKBONE_QWEN2_AUDIO:
        return build_qwen2_audio_student(model_dir, device=device)
    if backbone != BACKBONE_QWEN25_OMNI:
        raise ValueError(f"Unknown Ours backbone: {backbone}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = load_omni_processor(model_dir)
    model = load_omni_thinker(model_dir, dtype=dtype)
    freeze_non_lora_modules(model)
    student = attach_attention_lora(model, backbone=BACKBONE_QWEN25_OMNI)
    student.to(device)
    student.train()
    return processor, student


def save_lora_checkpoint(student: Any, output_dir: Path) -> None:
    """Save only the LoRA adapter weights."""

    output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(output_dir))
