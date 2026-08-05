import os
import gc
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file as load_sft
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from ideogram4.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS, IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, SEQUENCE_PADDING_INDICATOR
from ideogram4.latent_norm import get_latent_norm
from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals
from ideogram4.quantized_loading import FP8_TEXT_ENCODER_CONFIG_FLAG, Fp8Linear, FP8_WEIGHT_DTYPE, FP8_SCALE_SUFFIX, load_fp8_state_dict, swap_linears_to_fp8

# ==========================================
# 1. AUTOMATIC PATH RESOLUTION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "ideogram-4-fp8")

TEXT_ENCODER_DIR = os.path.join(MODEL_DIR, "text_encoder")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")
TRANSFORMER_INDEX = os.path.join(MODEL_DIR, "transformer", "diffusion_pytorch_model.safetensors.index.json")
UNCOND_INDEX = os.path.join(MODEL_DIR, "unconditional_transformer", "diffusion_pytorch_model.safetensors.index.json")
VAE_PATH = os.path.join(MODEL_DIR, "vae", "diffusion_pytorch_model.safetensors")

if not os.path.exists(TRANSFORMER_INDEX):
    print("\n>> Model weights not found locally. Downloading from Hugging Face...")
    snapshot_download(
        repo_id="ideogram-ai/ideogram-4-fp8",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
else:
    print("\n>> Local model weights found! Skipping download check.")

# ==========================================
# 2. MEMORY MANAGEMENT & PROMPT UTILS
# ==========================================
def empty_cache_xpu():
    """Aggressively clear System RAM and Intel XPU VRAM."""
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

def ensure_json_caption(prompt_str: str) -> str:
    """Ensures input is formatted as valid Ideogram 4 JSON schema."""
    prompt_str = prompt_str.strip()
    if prompt_str.startswith("{") and prompt_str.endswith("}"):
        try:
            json.loads(prompt_str)
            return prompt_str
        except Exception:
            pass
    
    caption_dict = {
        "high_level_description": prompt_str,
        "style_description": {
            "aesthetics": "high quality, detailed, vibrant",
            "lighting": "soft studio lighting",
            "photo": "sharp focus",
            "medium": "photograph"
        },
        "compositional_deconstruction": {
            "background": "clean detailed environment",
            "elements": [{"type": "obj", "desc": prompt_str}]
        }
    }
    return json.dumps(caption_dict, separators=(",", ":"), ensure_ascii=False)

def resolve_meta_tensors(model, device):
    """Moves a meta model to empty device tensors and reconstructs critical buffers."""
    model.to_empty(device=device)
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and module.inv_freq is not None:
            dim = getattr(module, "head_dim", getattr(module, "dim", None))
            if dim is None and len(module.inv_freq.shape) > 0:
                dim = module.inv_freq.shape[0] * 2
                
            if dim is not None:
                base = getattr(module, "base", getattr(module, "rope_theta", None))
                if base is None:
                    base = 5000000.0 if "Ideogram4" in module.__class__.__name__ else 1000000.0
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                module.inv_freq = inv_freq

def load_sharded_or_single_sft(index_path: str) -> dict[str, torch.Tensor]:
    """Loads safetensors directly into memory using mmap."""
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        shard_dir = os.path.dirname(index_path)
        shards = sorted(set(idx["weight_map"].values()))
        state_dict = {}
        for s in shards:
            state_dict.update(load_sft(os.path.join(shard_dir, s)))
        return state_dict
    
    single_file = index_path.removesuffix(".index.json")
    if os.path.exists(single_file):
        return load_sft(single_file)
    raise FileNotFoundError(f"Cannot find weights at {index_path} or {single_file}")

# ==========================================
# 3. THE HOT-SWAP ENGINE (OOM Prevention)
# ==========================================
class HotSwapTransformer:
    """
    Maintains a single 0-byte structural skeleton on the GPU. 
    It explicitly clears old tensor memory before pointing the model 
    to the new safetensor chunks to enforce a strict 10GB ceiling limit.
    """
    def __init__(self, config, index_path, device, dtype):
        self.device = device
        self.dtype = dtype
        
        # 1. Build 0-byte framework
        with torch.device("meta"):
            self.model = Ideogram4Transformer(config)
        
        # 2. Identify FP8 layers directly from the JSON map without loading weights
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        fp8_keys = set(idx["weight_map"].keys())
        
        self._swap_linears_to_fp8_keys(self.model, fp8_keys, compute_dtype=dtype)
        resolve_meta_tensors(self.model, torch.device(device))
        self.model.eval()

    def _swap_linears_to_fp8_keys(self, module, fp8_keys, compute_dtype, prefix=""):
        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}{name}"
            if isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in fp8_keys:
                setattr(module, name, Fp8Linear(
                    child.in_features, child.out_features, bias=child.bias is not None, compute_dtype=compute_dtype
                ))
            else:
                self._swap_linears_to_fp8_keys(child, fp8_keys, compute_dtype, prefix=f"{child_prefix}.")

    def load_weights(self, index_path):
        self.clear_weights() # Guarantee no overlap
        state_dict = load_sharded_or_single_sft(index_path)
        
        # Hot-plug tensor references directly to avoid deepcopy memory spiking
        for k, v in state_dict.items():
            target_module_path, _, param_name = k.rpartition(".")
            target_module = self.model.get_submodule(target_module_path) if target_module_path else self.model
            
            if v.dtype == FP8_WEIGHT_DTYPE:
                v_dev = v.to(device=self.device)
            elif k.endswith(FP8_SCALE_SUFFIX):
                v_dev = v.to(device=self.device, dtype=torch.float32)
            elif v.is_floating_point():
                v_dev = v.to(device=self.device, dtype=self.dtype)
            else:
                v_dev = v.to(device=self.device)
            
            if hasattr(target_module, "_parameters") and param_name in target_module._parameters:
                if isinstance(target_module._parameters[param_name], nn.Parameter):
                    target_module._parameters[param_name] = nn.Parameter(v_dev, requires_grad=False)
                else:
                    target_module._parameters[param_name] = v_dev
            elif hasattr(target_module, "_buffers") and param_name in target_module._buffers:
                target_module._buffers[param_name] = v_dev
        
        del state_dict
        empty_cache_xpu()
        
    def clear_weights(self):
        # Force overwrite major tensors back to meta device
        for module in self.model.modules():
            for param_name in ["weight", "bias", "weight_scale"]:
                if hasattr(module, "_parameters") and param_name in module._parameters and module._parameters[param_name] is not None:
                    module._parameters[param_name] = torch.tensor([], device="meta")
                if hasattr(module, "_buffers") and param_name in module._buffers and module._buffers[param_name] is not None:
                    module._buffers[param_name] = torch.tensor([], device="meta")
        empty_cache_xpu()

# ==========================================
# 4. MAIN GENERATION PIPELINE
# ==========================================
def detect_device():
    print("\n>> [0/3] Device detection and information")
    print(f"   - PyTorch version: {torch.__version__}")
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    selected = "xpu" if has_xpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - XPU available: {has_xpu}")
    print(f"   - Selected device: {selected}")
    return selected

def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    num_steps: int = 48,
    output_path: str = None
):
    device = "xpu"
    dtype = torch.bfloat16
    json_prompt = ensure_json_caption(prompt)

    print(f"\n--- Starting Generation (seed={seed}, steps={num_steps}) ---")

    # --- PHASE A: Text Encoding ---
    print("\n>> [1/3] Loading Text Encoder (Qwen3-VL)...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    
    cfg_path = os.path.join(TEXT_ENCODER_DIR, "config.json")
    with open(cfg_path) as f:
        cfg_data = json.load(f)
    
    if cfg_data.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False):
        config = AutoConfig.from_pretrained(TEXT_ENCODER_DIR, local_files_only=True)
        with torch.device("meta"):
            text_encoder = AutoModel.from_config(config, trust_remote_code=True)
        
        te_sd = load_sharded_or_single_sft(os.path.join(TEXT_ENCODER_DIR, "model.safetensors.index.json"))
        swap_linears_to_fp8(text_encoder, te_sd, compute_dtype=dtype)
        resolve_meta_tensors(text_encoder, torch.device(device))
        load_fp8_state_dict(text_encoder, te_sd, device=torch.device(device), dtype=dtype, assign=True, strict=False)
        del te_sd
    else:
        text_encoder = AutoModel.from_pretrained(TEXT_ENCODER_DIR, torch_dtype=dtype, local_files_only=True).to(device)
    
    text_encoder.eval()

    print(">> Processing prompt embeddings...")
    messages = [{"role": "user", "content": [{"type": "text", "text": json_prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])

    patch = 16
    grid_h, grid_w = height // patch, width // patch
    num_image_tokens = grid_h * grid_w
    total_seq_len = num_text_tokens + num_image_tokens

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    input_tokens = torch.zeros(1, total_seq_len, dtype=torch.long)
    text_pos_ids = torch.zeros(1, total_seq_len, 3, dtype=torch.long)
    position_ids = torch.zeros(1, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.full((1, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    indicator = torch.zeros(1, total_seq_len, dtype=torch.long)

    input_tokens[0, :num_text_tokens] = token_ids
    text_pos = torch.arange(num_text_tokens)
    text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
    text_pos_ids[0, :num_text_tokens] = text_pos_3d
    position_ids[0, :num_text_tokens] = text_pos_3d
    position_ids[0, num_text_tokens:] = image_pos
    indicator[0, :num_text_tokens] = LLM_TOKEN_INDICATOR
    indicator[0, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR
    segment_ids[0, :num_text_tokens + num_image_tokens] = 1

    input_tokens = input_tokens.to(device)
    text_pos_ids = text_pos_ids.to(device)
    position_ids = position_ids.to(device)
    segment_ids = segment_ids.to(device)
    indicator = indicator.to(device)

    with torch.no_grad():
        attn_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
        pos_2d = text_pos_ids[..., 0].contiguous()
        lm = text_encoder.language_model
        inputs_embeds = lm.embed_tokens(input_tokens)
        pos_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        
        position_embeddings = lm.rotary_emb(inputs_embeds, pos_4d[1:])
        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured = {}
        hs = inputs_embeds
        for idx, layer in enumerate(lm.layers):
            hs = layer(hs, attention_mask=None, position_ids=pos_4d[0], position_embeddings=position_embeddings)
            if idx in tap_set:
                captured[idx] = hs
        
        selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
        stacked = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(1, total_seq_len, -1)
        llm_features = (stacked * attn_mask.to(stacked.dtype).unsqueeze(-1)).to(torch.float32)

    print(">> Unloading Text Encoder to free VRAM...")
    del text_encoder, tokenizer, lm, captured, selected, stacked
    empty_cache_xpu()

    # --- PHASE B: ODE Interleaved Denoising (Hot-Swap Mode) ---
    print("\n>> [2/3] Denoising (Hot-Swapping weights interleaved to conserve VRAM)...")
    config = Ideogram4Config()
    transformer_wrapper = HotSwapTransformer(config, TRANSFORMER_INDEX, device, dtype)
    
    schedule = get_schedule_for_resolution((height, width), known_mean=0.0, std=1.5)
    step_intervals = make_step_intervals(num_steps) 
    
    gw_per_step = torch.full((num_steps,), 7.0, dtype=torch.float32, device=device)
    gw_per_step[-3:] = 3.0  # Polish steps

    neg_position_ids = position_ids[:, num_text_tokens:]
    neg_segment_ids = segment_ids[:, num_text_tokens:]
    neg_indicator = indicator[:, num_text_tokens:]
    neg_llm_features = torch.zeros(1, num_image_tokens, llm_features.shape[-1], dtype=llm_features.dtype, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    z = torch.randn(1, num_image_tokens, config.in_channels, dtype=torch.float32, device=device, generator=generator)
    text_z_padding = torch.zeros(1, num_text_tokens, config.in_channels, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(num_steps - 1, -1, -1):
            t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
            s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
            t = torch.full((1,), t_val, dtype=torch.float32, device=device)
            
            print(f"   -> Step {num_steps - i}/{num_steps}: Injecting Conditional Weights")
            transformer_wrapper.load_weights(TRANSFORMER_INDEX)
            
            pos_z = torch.cat([text_z_padding, z], dim=1)
            pos_v = transformer_wrapper.model(
                llm_features=llm_features, x=pos_z, t=t,
                position_ids=position_ids, segment_ids=segment_ids, indicator=indicator
            )[:, num_text_tokens:]
            
            print(f"   -> Step {num_steps - i}/{num_steps}: Injecting Unconditional Weights")
            transformer_wrapper.load_weights(UNCOND_INDEX)
            
            neg_v = transformer_wrapper.model(
                llm_features=neg_llm_features, x=z, t=t,
                position_ids=neg_position_ids, segment_ids=neg_segment_ids, indicator=neg_indicator
            )

            # Mathematical ODE Step
            gw_i = gw_per_step[i]
            v = gw_i * pos_v + (1.0 - gw_i) * neg_v
            z = z + v * (s_val - t_val)

    transformer_wrapper.clear_weights()
    del transformer_wrapper
    empty_cache_xpu()

    # --- PHASE C: VAE Decoding ---
    print("\n>> [3/3] Loading Autoencoder...")
    ae = AutoEncoder(AutoEncoderParams())
    ae_sd = convert_diffusers_state_dict(load_sft(VAE_PATH))
    ae.load_state_dict(ae_sd, assign=True)
    ae = ae.to(device=device, dtype=dtype)
    ae.eval()

    print(">> Decoding latents to pixels...")
    shift, scale = get_latent_norm()
    shift, scale = shift.to(device), scale.to(device)
    
    z = z * scale + shift
    ae_channels = z.shape[-1] // 4
    z = z.view(1, grid_h, grid_w, 2, 2, ae_channels).permute(0, 5, 1, 3, 2, 4).contiguous().view(1, ae_channels, grid_h * 2, grid_w * 2).to(dtype)

    with torch.no_grad():
        decoded = ae.decoder(z).float().clamp(-1.0, 1.0)
        decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    image = Image.fromarray(decoded[0])

    del ae, ae_sd
    empty_cache_xpu()

    # --- PHASE D: Save Output ---
    output_dir = os.path.join(SCRIPT_DIR, "output", "ideogram4_fp8")
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"ideogram4_{timestamp}_{seed}.png")

    image.save(output_path)
    print(f"\n>> Success! Image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=48)
    parser.add_argument("--quantization", type=str, default="fp8")
    parser.add_argument("--no-magic-prompt", action="store_true")
    args, _ = parser.parse_known_args()

    detect_device()

    if args.prompt:
        user_prompt = args.prompt
        seed = args.seed if args.seed != 0 else np.random.randint(0, 2**31 - 1)
        steps = args.num_steps
    else:
        user_prompt = input("\nEnter prompt: ").strip()
        if not user_prompt:
            user_prompt = "A detailed shot of an Intel Arc A770 graphics card sitting on a desk."
            print(f"Using default prompt: {user_prompt}")
        
        steps_input = input("Enter steps [default 48]: ").strip() or "48"
        steps = int(steps_input)
        
        seed_input = input("Enter seed (leave blank for random): ").strip()
        seed = int(seed_input) if seed_input else np.random.randint(0, 2**31 - 1)

    generate_image(user_prompt, seed=seed, num_steps=steps, width=args.width, height=args.height)