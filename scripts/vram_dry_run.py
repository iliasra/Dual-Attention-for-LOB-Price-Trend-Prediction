from __future__ import annotations

import argparse
from dataclasses import replace

import torch
import torch.nn.functional as F

from compatibility import autocast_context, make_grad_scaler, torch_device_type
from configuration import load_config
from model import build_model


def bytes_to_gib(x: int) -> float:
    return x / 1024**3


def run_vram_test(
    batch_size: int,
    sequence_length: int,
    num_features: int,
    steps: int = 5,
    device: str = "cuda",
) -> None:
    device_obj = torch.device(device)
    device_type = torch_device_type(device_obj)
    if device_type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = load_config()
    model_config = replace(
        config.model,
        d_input=num_features,
        max_dt=1.0,  # dry-run: placeholder value
    )
    training_config = config.training

    amp_enabled = bool(training_config.use_amp and device_type == "cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_obj)

    model = build_model(model_config).to(device_obj)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    scaler = make_grad_scaler(device=device_obj, enabled=amp_enabled)

    x = torch.randn(batch_size, sequence_length, num_features, device=device_obj)
    t = torch.arange(sequence_length, device=device_obj, dtype=torch.float32)
    t = t.unsqueeze(0).repeat(batch_size, 1)
    y = torch.randint(
        low=0,
        high=model_config.num_classes,
        size=(batch_size,),
        device=device_obj,
    )

    print("VRAM dry-run configuration")
    print(f"B={batch_size}, T={sequence_length}, F={num_features}")
    print(f"d_model={model_config.d_model}")
    print(f"feature_embed_dim={model_config.feature_embed_dim}")
    print(f"num_heads={model_config.num_heads}")
    print(f"AMP enabled={amp_enabled}")
    print()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device=device_obj, enabled=amp_enabled):
            logits = model(x, t)
            loss = F.cross_entropy(logits, y)
            moe_loss = getattr(model, "moe_load_balancing_loss", None)
            if moe_loss is not None:
                loss = loss + moe_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=training_config.grad_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize(device_obj)

        allocated = torch.cuda.memory_allocated(device_obj)
        reserved = torch.cuda.memory_reserved(device_obj)
        peak_allocated = torch.cuda.max_memory_allocated(device_obj)
        peak_reserved = torch.cuda.max_memory_reserved(device_obj)

        print(
            f"step={step + 1:02d} "
            f"allocated={bytes_to_gib(allocated):.2f} GiB "
            f"reserved={bytes_to_gib(reserved):.2f} GiB "
            f"peak_allocated={bytes_to_gib(peak_allocated):.2f} GiB "
            f"peak_reserved={bytes_to_gib(peak_reserved):.2f} GiB"
        )

    free, total = torch.cuda.mem_get_info(device_obj)
    print()
    print(f"GPU total={bytes_to_gib(total):.2f} GiB")
    print(f"GPU free after test={bytes_to_gib(free):.2f} GiB")
    print(f"Peak allocated={bytes_to_gib(torch.cuda.max_memory_allocated(device_obj)):.2f} GiB")
    print(f"Peak reserved={bytes_to_gib(torch.cuda.max_memory_reserved(device_obj)):.2f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--num-features", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    run_vram_test(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_features=args.num_features,
        steps=args.steps,
    )
