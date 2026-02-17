import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from sonics.models.hf_model import HFAudioClassifier
from sonics.utils.dataset import get_dataloader
from sonics.utils.metrics import get_part_result
from sonics.utils.losses import BCEWithLogitsLoss, SigmoidFocalLoss
from sonics.utils.seed import set_seed, worker_init_fn

# Reuse the evaluation loop from training
from train import valid_loop


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark pretrained SONICS models on a local test split."
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of HF model IDs or local paths",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="Path to test CSV (default: test.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (default: use config)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers (default: 0 to avoid worker stalls)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/benchmarks",
        help="Directory to save benchmark outputs",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    if not model_ids:
        raise ValueError("No models provided. Use --models with at least one model ID.")

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"> Using device: {device}")

    # Load test data once
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)

    for model_id in model_ids:
        print(f"\n=== Benchmarking: {model_id} ===")

        # Load model from HF or local path
        model = HFAudioClassifier.from_pretrained(
            model_id, map_location=str(device), strict=False
        )
        model.to(device)
        model.eval()

        cfg = model.config

        # Seed
        seed = getattr(cfg.environment, "seed", 42) if hasattr(cfg, "environment") else 42
        set_seed(seed)

        # Batch size
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else getattr(cfg.validation, "batch_size", 64)
        )

        # Dataloader
        test_dataloader = get_dataloader(
            test_df.filepath.tolist(),
            test_df.target.tolist(),
            skip_times=test_df.skip_time.tolist() if cfg.audio.skip_time else None,
            max_len=cfg.audio.max_len,
            batch_size=batch_size,
            num_classes=getattr(cfg, "num_classes", 1),
            train=False,
            random_sampling=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=None,
            distributed=False,
        )

        # Loss
        loss_name = getattr(cfg.loss, "name", "BCEWithLogitsLoss")
        if loss_name == "BCEWithLogitsLoss":
            criterion = BCEWithLogitsLoss(
                label_smoothing=getattr(cfg.loss, "label_smoothing", 0.0)
            )
        elif loss_name == "SigmoidFocalLoss":
            criterion = SigmoidFocalLoss(
                alpha=getattr(cfg.loss, "alpha", 0.25),
                gamma=getattr(cfg.loss, "gamma", 2.0),
                label_smoothing=getattr(cfg.loss, "label_smoothing", 0.0),
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Eval
        (
            test_loss,
            test_acc,
            test_f1,
            test_sens,
            test_spec,
            test_pred_df,
        ) = valid_loop(model, test_dataloader, criterion, device, cfg, desc="Test")

        # Prepare output paths
        model_slug = model_id.replace("/", "__").replace(":", "__")
        out_dir = Path(args.output_dir) / model_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        test_pred_df = pd.concat([test_df, test_pred_df], axis=1)
        pred_path = out_dir / "test_predictions.csv"
        test_pred_df.to_csv(pred_path, index=False)

        # Partition results (only if required columns exist)
        required_cols = {"artist_overlap", "label", "duration", "algorithm"}
        if required_cols.issubset(set(test_pred_df.columns)):
            part_result_df, _ = get_part_result(test_pred_df)
            part_path = out_dir / "test_partition_results.csv"
            part_result_df.to_csv(part_path, index=False)
            print(f"> Saved partition results: {part_path}")
        else:
            missing = sorted(required_cols - set(test_pred_df.columns))
            part_path = None
            print(
                f"> Skipping partition results (missing columns): {', '.join(missing)}"
            )

        # Summary
        summary = {
            "model": model_id,
            "loss": test_loss,
            "acc": test_acc,
            "f1": test_f1,
            "sens": test_sens,
            "spec": test_spec,
        }
        summary_df = pd.DataFrame([summary])
        summary_path = out_dir / "test_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("> Summary:")
        print(summary_df.to_markdown(index=False, tablefmt="grid"))
        print(f"> Saved predictions: {pred_path}")
        if part_path is not None:
            print(f"> Saved partition results: {part_path}")
        print(f"> Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
