"""Cloud model: projection -> Llama (causal mask removed, LoRA on q/v) -> linear head.

Needs: torch, transformers, peft (and bitsandbytes for 4-bit QLoRA).
"""
import torch
import torch.nn as nn


def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def disable_causal(module):
    """Turn off the causal flag on every attention layer (belt-and-braces with the 4D mask)."""
    for m in module.modules():
        if hasattr(m, "is_causal"):
            try:
                m.is_causal = False
            except Exception:
                pass


def full_attention_mask(batch, length, dtype, device, key_padding=None):
    """Additive 4D mask (B, 1, L, L): 0 = attend. Every flow sees every other flow.

    key_padding: optional (B, L) bool, True where a position is padding (never attended).
    """
    mask = torch.zeros(batch, 1, length, length, dtype=dtype, device=device)
    if key_padding is not None:
        neg = torch.finfo(dtype).min
        mask = mask.masked_fill(key_padding[:, None, None, :], neg)
    return mask


class DELLMCloud(nn.Module):
    def __init__(self, backbone="meta-llama/Llama-3.1-8B", token_dim=768, n_classes=8,
                 use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 dtype="bfloat16", load_in_4bit=False, no_llm=False,
                 gradient_checkpointing=True, hidden_size_if_no_llm=4096):
        super().__init__()
        self.no_llm = no_llm
        self.dtype = _dtype(dtype)
        if no_llm:
            self.backbone = None
            hidden = hidden_size_if_no_llm
        else:
            from transformers import AutoModel
            kwargs = {"torch_dtype": self.dtype, "attn_implementation": "sdpa"}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True)
                kwargs["device_map"] = {"": 0}
            base = AutoModel.from_pretrained(backbone, **kwargs)
            hidden = base.config.hidden_size
            disable_causal(base)
            if load_in_4bit:
                from peft import prepare_model_for_kbit_training
                base = prepare_model_for_kbit_training(
                    base, use_gradient_checkpointing=gradient_checkpointing)
            elif gradient_checkpointing:
                base.gradient_checkpointing_enable()
                if hasattr(base, "enable_input_require_grads"):
                    base.enable_input_require_grads()
            for p in base.parameters():
                p.requires_grad = False
            if use_lora:
                from peft import LoraConfig, get_peft_model
                cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 target_modules=["q_proj", "v_proj"], bias="none")
                base = get_peft_model(base, cfg)
            self.backbone = base
        self.proj = nn.Sequential(nn.Linear(token_dim, hidden), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, tokens, key_padding=None):
        """tokens: (B, L, token_dim) -> logits (B, L, n_classes)."""
        e = self.proj(tokens.float())
        if self.no_llm:
            return self.head(e)
        e = e.to(self.dtype)
        B, L, _ = e.shape
        mask = full_attention_mask(B, L, self.dtype, e.device, key_padding)
        out = self.backbone(inputs_embeds=e, attention_mask=mask, use_cache=False)
        return self.head(out.last_hidden_state.float())

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def merge_lora(self):
        """Fold B*A into W after training (no inference cost afterwards)."""
        if self.backbone is not None and hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload()
            disable_causal(self.backbone)


@torch.no_grad()
def check_bidirectional(model, token_dim=768, length=64, device="cpu"):
    """Mask test: changing the LAST flow must change the FIRST flow's output.

    Returns True if attention is bidirectional (causal mask really removed).
    """
    if model.no_llm:
        return True
    model.eval()
    x = torch.randn(1, length, token_dim, device=device)
    y1 = model(x)[0, 0]
    x2 = x.clone()
    x2[0, -1] += 5.0
    y2 = model(x2)[0, 0]
    return bool((y1 - y2).abs().max().item() > 1e-4)


def save_cloud(model, path):
    """Save projection + head; LoRA adapter (if any) is saved next to it."""
    import os
    os.makedirs(path, exist_ok=True)
    torch.save({"proj": model.proj.state_dict(), "head": model.head.state_dict()},
               os.path.join(path, "proj_head.pt"))
    if model.backbone is not None and hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(os.path.join(path, "lora_adapter"))


def load_cloud(path, **model_kwargs):
    """Rebuild the model and load trained projection/head and LoRA adapter."""
    import os
    use_lora = model_kwargs.pop("use_lora", True)
    model = DELLMCloud(use_lora=False, **model_kwargs)
    adapter = os.path.join(path, "lora_adapter")
    if use_lora and model.backbone is not None and os.path.isdir(adapter):
        from peft import PeftModel
        model.backbone = PeftModel.from_pretrained(model.backbone, adapter)
        disable_causal(model.backbone)
    sd = torch.load(os.path.join(path, "proj_head.pt"), map_location="cpu")
    model.proj.load_state_dict(sd["proj"])
    model.head.load_state_dict(sd["head"])
    return model
