from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark token-wise chunk training throughput.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-window", type=int, default=None)
    parser.add_argument("--d-input", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    device = torch.device(args.device or config.training.device)
    sequence_window = int(args.sequence_window or config.data.sequence_window)
    batch_size = int(args.batch_size or config.training.batch_size)
    d_input = args.d_input or config.model.d_input
    if d_input is None:
        d_input = 64
        print(f"Using synthetic d_input={d_input}; pass --d-input to override.")
    if config.model.max_dt is None:
        config.model.max_dt = 1e9

    model = build_model(config.model, d_input=int(d_input)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = bool(config.training.use_amp and device.type == "cuda")
    autocast_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else None
    supervised_tail = sequence_window
    if config.training.sequence_supervision.token_chunk_enabled:
        warmup = int(config.training.sequence_supervision.loss_warmup_tokens or 0)
        if warmup >= sequence_window:
            raise ValueError(
                "training.sequence_supervision.loss_warmup_tokens must be smaller than "
                "--sequence-window for the benchmark."
            )
        supervised_tail = sequence_window - warmup
    mask = torch.zeros((batch_size, sequence_window), dtype=torch.bool, device=device)
    mask[:, -supervised_tail:] = True

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_seconds = 0.0
    total_tokens = 0
    total_labels = 0
    for step in range(args.warmup + args.steps):
        x = torch.randn((batch_size, sequence_window, int(d_input)), device=device)
        t = torch.arange(sequence_window, device=device, dtype=torch.float32).repeat(batch_size, 1)
        y = torch.randint(0, config.model.num_classes, (batch_size, sequence_window), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=autocast_dtype,
        ):
            logits = model(x, t, tokenwise=config.training.sequence_supervision.token_chunk_enabled)
            if logits.ndim == 3:
                loss = criterion(logits[mask], y[mask])
                labels = int(mask.sum().item())
            else:
                loss = criterion(logits, y[:, -1])
                labels = batch_size
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        if step >= args.warmup:
            total_seconds += elapsed
            total_tokens += batch_size * sequence_window
            total_labels += labels

    peak_gib = None
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(
        "benchmark_token_chunk: "
        f"batch_size={batch_size}, sequence_window={sequence_window}, d_input={d_input}, "
        f"steps={args.steps}, seconds={total_seconds:.4f}, "
        f"tokens_per_s={total_tokens / max(total_seconds, 1e-12):.2f}, "
        f"labels_per_s={total_labels / max(total_seconds, 1e-12):.2f}, "
        f"peak_cuda_gib={peak_gib if peak_gib is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
