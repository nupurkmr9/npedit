import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image

from models.latent_fm import VelocityModel
from models.latent_fm_factory import create_latent_fm
from trainers import load_config
from utils.fsdp import fwd_only_mode
from utils_fm.sampler import FlowSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/zimage_inference.yaml")
    parser.add_argument("--ckpt", type=str, default="experiments/zimage_npedit_internvl.pt")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)    
    return parser.parse_args()


def init_single_process_group() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0,
            device_id=torch.device("cuda:0"),
        )


def preprocess_reference(image_path: str, resolution: int) -> Image.Image:
    """Mirror data/instruct.py:109-124 non-bucket path: resize so h=resolution,
    width scaled and rounded to nearest multiple of 16, LANCZOS."""
    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size
    new_h = resolution
    new_w = int(w * new_h / h)
    new_w = max(16, round(new_w / 16) * 16)
    return pil.resize((new_w, new_h), Image.LANCZOS)


def main() -> None:
    args = parse_args()

    init_single_process_group()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = load_config(args.config)
    assert "student" in config, "Expected config['student'] section (see configs/zimage_inference.yaml)"
    model_cfg = config["student"]["model"]
    model_cfg["denoiser"].pop("fsdp", None)
    model_cfg["text_encoder"].pop("fsdp", None)
    model_cfg["vae"].pop("fsdp", None)

    print("Creating LatentFM components...")
    latent_fm = create_latent_fm(config["student"], device, create_ema=False, mode="student")
    assert latent_fm.denoiser is not None
    assert latent_fm.vae is not None
    assert latent_fm.text_encoder is not None

    print(f"Loading denoiser weights from {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = latent_fm.denoiser.load_state_dict(state_dict, strict=True)
    print(f"  loaded {len(state_dict)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    latent_fm.denoiser.eval()

    pil = preprocess_reference(args.image, args.resolution)
    img = T.ToTensor()(pil)  # (3, H, W) in [0, 1]
    img = img.unsqueeze(0).unsqueeze(2).to(device)  # (1, 3, 1, H, W)
    img = img * 2.0 - 1.0

    timesteps = torch.tensor([1.0, 0.9474, 0.8571, 0.6667], dtype=torch.float32, device=device)

    with torch.no_grad(), fwd_only_mode(latent_fm.denoiser):
        generator = torch.Generator(device=device).manual_seed(args.seed)

        with fwd_only_mode(latent_fm.vae):
            reference_img_clean = latent_fm.vae.encode(img)
        _, _, latent_f, latent_h, latent_w = reference_img_clean.shape

        with fwd_only_mode(latent_fm.text_encoder):
            txt, txt_datum_lens, txt_embedding_mask = latent_fm.text_encoder([args.instruction])

        noise = torch.randn(
            1, 16, latent_f, latent_h, latent_w,
            device=device, dtype=torch.float32, generator=generator,
        )
        img_datum_lens = torch.full(
            (1,), latent_f * (latent_h // 2) * (latent_w // 2),
            device=device, dtype=torch.int32,
        ) * 2  # doubled — reference latent is concatenated in VelocityModel

        velocity_model = VelocityModel(
            denoiser=latent_fm.denoiser,
            txt=txt,
            txt_datum_lens=txt_datum_lens,
            txt_embedding_mask=txt_embedding_mask,
            cfg_scale=1.0,
            reference_img_clean=reference_img_clean,
            energy_preserve_cfg=False,
        )

        sampler = FlowSampler(
            velocity_model=velocity_model,
            noiser=latent_fm.flow_noiser,
            t_warper=latent_fm.time_warper,
            sample_method="ddim",
            min_timestep=0.0,
        )

        print(f"Sampling")
        latents, _ = sampler(
            x=noise,
            x_datum_lens=img_datum_lens,
            num_steps=len(timesteps),
            warp_len=int(img_datum_lens[0].item()),
            rng=generator,
            eta=1.0,
            timesteps=timesteps,
        )

        with fwd_only_mode(latent_fm.vae):
            out = latent_fm.vae.decode(latents)

    out = ((out.float() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    out_np = out.permute(0, 2, 3, 4, 1)[0, 0].cpu().numpy()  # (H, W, 3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_np).save(args.output)
    print(f"Wrote {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()