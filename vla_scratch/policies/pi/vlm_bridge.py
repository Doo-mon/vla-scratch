import importlib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp._fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
    FSDPModule,
    register_fsdp_forward_method,
)
import einops

from vla_scratch.policies.pi.utils import (
    make_att_2d_masks,
    attention_fill_false_to_inf,
)
from vla_scratch.policies.utils import apply_rotary_pos_emb


def _gemma_decoder_layer_custom_forward(
    self, hidden_states, prefix_att_mask, position_embeddings
):
    """Custom forward for a GemmaDecoderLayer used in prefix encoding.

    This mirrors the previous inline `compute_layer` function, but is defined
    as a bound method that attaches to `GemmaDecoderLayer` as `custom_forward`.
    """
    pre_att = self.input_layernorm(hidden_states)
    input_shape = hidden_states.shape[:-1]  # [batch_size, seq_len]
    head_shape = (*input_shape, -1, self.self_attn.head_dim)

    # attention
    torch.cuda.nvtx.range_push("project_qkv")
    q = self.self_attn.q_proj(pre_att).view(head_shape)
    k = self.self_attn.k_proj(pre_att).view(head_shape)
    v = self.self_attn.v_proj(pre_att).view(head_shape)
    q = einops.rearrange(q, "b seq head dim -> b head seq dim")
    k = einops.rearrange(k, "b seq head dim -> b head seq dim")
    v = einops.rearrange(v, "b seq head dim -> b head seq dim")
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("rotary_embedding")
    cos, sin = position_embeddings
    q_rotate, k_rotate = apply_rotary_pos_emb(q, k, cos, sin)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("attention")
    out_att = F.scaled_dot_product_attention(
        q_rotate,
        k_rotate,
        v,
        attn_mask=prefix_att_mask,
        scale=self.self_attn.scaling,
    )
    out_att = einops.rearrange(
        out_att, "b head seq dim -> b seq (head dim)"
    ).contiguous()
    out_att = self.self_attn.o_proj(out_att)
    res_att = hidden_states + out_att
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("mlp")
    pre_mlp = self.post_attention_layernorm(res_att)
    out_mlp = self.mlp(pre_mlp)
    res_mlp = res_att + out_mlp
    torch.cuda.nvtx.range_pop()
    return res_mlp, (k_rotate, v)


class VLMBridge(nn.Module):
    """Abstract base class for VLM bridges.

    Responsibilities:
      - Handle model-specific preprocessing (tokenization/vision).
      - Run the VLM transformer forward in a layer-wise loop with optional checkpointing.
      - Return (hidden_states, prefix_pad_masks, kv_cache_list).
    """

    layer_custom_forward_name = "forward"

    def get_text_dims(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    def encode_prefix(
        self,
        *,
        images: torch.Tensor,
        image_masks: torch.Tensor,
        tasks: List[str],
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)


class PaligemmaBridge(VLMBridge):
    layer_custom_forward_name = "custom_forward"

    def __init__(self, *, model_id: str, vlm_type: str, max_length: int = 64):
        super().__init__()
        self.max_length = max_length

        # Instantiate model
        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(f"transformers has no class named '{vlm_type}'.") from e

        device_map = torch.cuda.current_device() if torch.cuda.is_available() else None
        from transformers import PaliGemmaForConditionalGeneration

        self.model: PaliGemmaForConditionalGeneration = vlm_cls.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device_map,
        )

        # Tokenizer for text prompts
        PaliGemmaProcessor = getattr(tfm, "PaliGemmaProcessor")
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        # Attach custom forward to Gemma decoder layers
        from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

        setattr(
            GemmaDecoderLayer,
            self.layer_custom_forward_name,
            _gemma_decoder_layer_custom_forward,
        )

    def get_text_dims(self) -> Tuple[int, int, int]:
        cfg = self.model.config.text_config
        return (
            cfg.num_hidden_layers,
            cfg.head_dim,
            cfg.num_key_value_heads,
            cfg.hidden_size,
        )

    def _tokenize(
        self, tasks: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts = [f"<bos>Task: {t}; \n Action:" for t in tasks]
        encoded = self.tokenizer(
            prompts,
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        # attention mask should be 1 for text and 0 for padding
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    def encode_prefix(
        self,
        *,
        images: torch.Tensor,
        image_masks: torch.Tensor,
        tasks: List[str],
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        device = images.device

        # Tokenize text prompts -> ids and masks
        input_ids, attn_mask = self._tokenize(tasks, device)
        input_ids = input_ids.long()
        lang_masks = attn_mask.bool()

        # Image embeddings (flatten camera dimension)
        b, n_cam = images.shape[0], images.shape[1]
        images_flat = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_emb_flat = self._apply_checkpoint(
            self.model.model.get_image_features, images_flat
        )
        img_emb = einops.rearrange(img_emb_flat, "(b n) t d -> b (n t) d", b=b, n=n_cam)
        img_mask_repeat = einops.repeat(
            image_masks, "b n 1 -> b (n t)", t=img_emb_flat.shape[1]
        )

        # Text embeddings
        lang_emb = self._apply_checkpoint(
            self.model.language_model.embed_tokens, input_ids
        )

        # Concatenate with optional extra tokens (e.g., observation registers)
        embs = [img_emb, lang_emb]
        pad_masks = [img_mask_repeat, lang_masks]
        att_masks = [
            torch.zeros(img_emb.shape[1], dtype=torch.bool, device=device),
            torch.zeros(lang_emb.shape[1], dtype=torch.bool, device=device),
        ]

        if extra_embs is not None:
            embs.append(extra_embs)
            pad_masks.append(extra_pad_masks)
            att_masks.append(extra_att_masks)

        embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.cat(att_masks, dim=0).expand(b, -1)

        # Build attention mask for prefix tokens
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks_1d)
        prefix_att_mask = attention_fill_false_to_inf(prefix_att_2d)[:, None, :, :]

        # Position embeddings for rotary attention
        lm = self.model.language_model
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        pos_emb = lm.rotary_emb.forward(embs, position_ids)

        # Layer-wise forward with per-layer KV capture
        hidden_states = embs * (embs.shape[-1] ** 0.5)
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in lm.layers:
            hidden_states, (k, v) = self._apply_checkpoint(
                getattr(layer, self.layer_custom_forward_name),
                hidden_states,
                prefix_att_mask,
                pos_emb,
            )
            kv_cache_list.append((k, v))

        hidden_states = lm.norm(hidden_states)
        return hidden_states, prefix_pad_masks, kv_cache_list


def _qwen_text_decoder_layer_custom_forward(
    self,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
):
    # Mirror Qwen3VLTextDecoderLayer.forward but without DynamicCache.
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.self_attn.head_dim)

    # Projections with QK norm
    q = self.self_attn.q_proj(hidden_states).view(hidden_shape)
    q = self.self_attn.q_norm(q).transpose(1, 2)  # (b, h_q, L, d)
    k = self.self_attn.k_proj(hidden_states).view(hidden_shape)
    k = self.self_attn.k_norm(k).transpose(1, 2)  # (b, h_kv, L, d)
    v = (
        self.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    )  # (b, h_kv, L, d)

    # RoPE
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Repeat KV heads to match Q heads (GQA)
    bsz, h_q, L, d = q.shape
    h_kv = k.shape[1]
    rep = max(1, h_q // h_kv)
    if rep > 1:
        k_attn = (
            k[:, :, None, :, :].expand(bsz, h_kv, rep, L, d).reshape(bsz, h_q, L, d)
        )
        v_attn = (
            v[:, :, None, :, :].expand(bsz, h_kv, rep, L, d).reshape(bsz, h_q, L, d)
        )
    else:
        k_attn = k
        v_attn = v

    # Attention
    if attention_mask is not None:
        # attention_mask expected shape [b, 1, L, L] additive; expand over heads
        attn_mask = attention_mask.expand(bsz, h_q, L, L)
    else:
        attn_mask = None

    attn_out = F.scaled_dot_product_attention(
        q,
        k_attn,
        v_attn,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=self.self_attn.head_dim**-0.5,
    )
    attn_out = attn_out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_out = self.self_attn.o_proj(attn_out)

    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, (k, v)


class Qwen3VLBridge(VLMBridge):
    """Qwen3-VL bridge.

    Note: Full per-layer manual forward with KV capture requires detailed model internals.
    This class provides the interface and dimension metadata. Integrating the manual
    layer loop is left as a follow-up.
    """

    layer_custom_forward_name = "custom_forward"

    def __init__(self, *, model_id: str, vlm_type: str, max_length: int = 256):
        super().__init__()
        from transformers import AutoProcessor

        # Instantiate model
        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(f"transformers has no class named '{vlm_type}'.") from e

        device_map = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.model = vlm_cls.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map=device_map,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.max_length = max_length

        # Attach custom forward once to decoder layer class
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        setattr(
            Qwen3VLTextDecoderLayer,
            self.layer_custom_forward_name,
            _qwen_text_decoder_layer_custom_forward,
        )

    def get_text_dims(self) -> Tuple[int, int, int]:
        cfg = self.model.config.text_config
        # Prefer explicit head_dim if present; else derive
        head_dim = cfg.head_dim
        num_kv_heads = cfg.num_key_value_heads
        hidden = cfg.hidden_size
        return cfg.num_hidden_layers, head_dim, num_kv_heads, hidden

    def encode_prefix(
        self,
        *,
        images: torch.Tensor,
        image_masks: torch.Tensor,
        tasks: List[str],
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        device = images.device
        bsz, n_cam = images.shape[0], images.shape[1]

        # Build chat messages: one sample per batch with its N images then text
        # The processor will load visuals from these messages and align tokens/placeholders accordingly.
        batch_messages: List[List[dict]] = []
        # Convert images to a list of per-sample visuals (keep as tensors; processor supports tensors)
        # Note: images may be normalized to [-1, 1] by upstream transforms. Qwen processors accept tensors
        # in [0, 1] or [0, 255]. If input appears normalized, we map back to [0, 255] for safety.
        imgs = images
        if torch.is_floating_point(imgs):
            # Detect normalization approximately: values mostly within [-1, 1]
            if imgs.min() < -0.1 or imgs.max() > 1.1:
                # assume raw bytes scaling [0,255]; convert to 0-255 uint8 directly
                imgs_uint8 = imgs.clamp(0, 255).to(torch.uint8)
            else:
                # map [-1,1] to [0,255] or [0,1] to [0,255]
                if imgs.min() < 0.0:
                    imgs_uint8 = ((imgs + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
                else:
                    imgs_uint8 = (imgs * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            imgs_uint8 = imgs

        for b in range(bsz):
            content: List[dict] = []
            for c in range(n_cam):
                # Processor supports CHW PyTorch tensors; keep channel-first
                content.append({"type": "image", "image": imgs_uint8[b, c]})
            task = tasks[b] if isinstance(tasks, list) else tasks
            content.append({"type": "text", "text": task})
            batch_messages.append([{"role": "user", "content": content}])

        # Use processor to produce tokenized text and vision inputs aligned with placeholders
        feats = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            text_kwargs={
                "max_length": self.max_length,
                "truncation": True,
                "padding": "max_length",
                "return_tensors": "pt",
            },
        )

        # Move to correct device
        input_ids: torch.Tensor = feats["input_ids"].to(device)
        attention_mask: torch.Tensor = feats["attention_mask"].to(device)
        pixel_values: Optional[torch.Tensor] = feats.get("pixel_values")
        image_grid_thw: Optional[torch.Tensor] = feats.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        # Build embeddings and replace image placeholders with visual features
        lm = self.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)

        image_embeds = None
        deepstack_image_embeds = None
        if pixel_values is not None:
            image_embeds_list, deepstack_image_embeds = self.model.get_image_features(
                pixel_values, image_grid_thw=image_grid_thw
            )
            image_embeds = torch.cat(image_embeds_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # get_placeholder_mask is defined on the inner Qwen3VLModel
            image_mask, _ = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
        else:
            visual_pos_masks = None

        # Compute multimodal position_ids (3, b, seq) via model helper
        position_ids, _rope_deltas = self.model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        text_position_ids = position_ids[0]

        # Optionally append extra_embs (e.g., observation registers) as additional tokens at the end
        if extra_embs is not None:
            if extra_pad_masks is None:
                raise ValueError("extra_pad_masks must be provided with extra_embs")
            if extra_att_masks is None:
                # Not used by Qwen path, but keep api parity
                extra_att_masks = torch.zeros(
                    extra_embs.shape[1], dtype=torch.bool, device=device
                )

            # Concatenate embeddings
            inputs_embeds = torch.cat(
                [inputs_embeds, extra_embs.to(inputs_embeds.dtype)], dim=1
            )

            # Extend attention mask (treat extra tokens as non-padding)
            bsz = attention_mask.shape[0]
            extra_len = extra_embs.shape[1]
            extra_mask = torch.ones(
                (bsz, extra_len), dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

            # Extend position_ids by continuing the text axis; replicate across 3 planes for non-vision tokens
            # Compute last valid text position per batch (at last non-pad index)
            valid_lengths = attention_mask[:, :-extra_len].sum(
                dim=1
            )  # length before extras
            # Gather the last text position from the base text_position_ids
            gather_idx = (valid_lengths - 1).clamp(min=0)
            last_pos = text_position_ids[torch.arange(bsz, device=device), gather_idx]
            # Build incremental positions for the extras: last_pos + [1..extra_len]
            increments = torch.arange(
                1, extra_len + 1, device=device, dtype=last_pos.dtype
            ).unsqueeze(0)
            extra_text_pos = last_pos.unsqueeze(1) + increments
            # Replicate across 3 planes (T/H/W) for the additional tokens
            extra_pos_3d = extra_text_pos.unsqueeze(0).expand(3, -1, -1)
            position_ids = torch.cat([position_ids, extra_pos_3d], dim=-1)

            # Ensure any visual masks align in length (no visual embeddings for extras)
            if visual_pos_masks is not None:
                extra_visual = torch.zeros(
                    (bsz, extra_len), dtype=visual_pos_masks.dtype, device=device
                )
                visual_pos_masks = torch.cat([visual_pos_masks, extra_visual], dim=1)

        # Build a simple additive causal mask with padding: [B, 1, L, L]
        L = inputs_embeds.shape[1]
        seq_arange = torch.arange(L, device=device)
        causal = seq_arange.view(1, L, 1) >= seq_arange.view(1, 1, L)  # [1,L,L]
        pad = attention_mask.bool()  # [B,L]
        base = causal & pad[:, None, :] & pad[:, :, None]  # [B,L,L]
        attn_mask = attention_fill_false_to_inf(base)[:, None, :, :].type(
            inputs_embeds.dtype
        )

        # Precompute rotary embeddings once
        position_embeddings = lm.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Manual per-layer forward with checkpoint and DeepStack visual add-ins
        for layer_idx, decoder_layer in enumerate(lm.layers):
            hidden_states, (k, v) = self._apply_checkpoint(
                getattr(decoder_layer, self.layer_custom_forward_name),
                hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )

            # DeepStack: add early visual embeddings back into hidden states
            if deepstack_image_embeds is not None and layer_idx in range(
                len(deepstack_image_embeds)
            ):
                vmask = visual_pos_masks
                if vmask is not None:
                    local_this = hidden_states[
                        vmask, :
                    ].clone() + deepstack_image_embeds[layer_idx].to(
                        hidden_states.device, hidden_states.dtype
                    )
                    hidden_states[vmask, :] = local_this

            kv_cache_list.append((k, v))

        hidden_states = lm.norm(hidden_states)

        # Prefix pad masks match attention_mask boolean
        prefix_pad_masks = attention_mask.bool()

        return hidden_states, prefix_pad_masks, kv_cache_list
