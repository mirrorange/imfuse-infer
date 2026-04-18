"""imfuse-infer CLI — predict and convert subcommands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imfuse-infer",
        description="IM-Fuse brain tumor segmentation inference tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- predict ----
    p = sub.add_parser("predict", help="Run segmentation inference on NIfTI volumes")
    p.add_argument("--flair", type=str, default=None, help="Path to FLAIR (t2f) NIfTI")
    p.add_argument("--t1ce", type=str, default=None, help="Path to T1ce (t1c) NIfTI")
    p.add_argument("--t1", type=str, default=None, help="Path to T1 (t1n) NIfTI")
    p.add_argument("--t2", type=str, default=None, help="Path to T2 (t2w) NIfTI")
    p.add_argument("-o", "--output", type=str, required=True, help="Output segmentation NIfTI path")
    p.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth or .safetensors)")
    p.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    p.add_argument("--num-cls", type=int, default=4, help="Number of classes (default: 4)")
    p.add_argument("--no-interleaved", action="store_true", help="Disable interleaved tokenization")
    p.add_argument("--no-mamba-skip", action="store_true", help="Disable Mamba skip connections")
    p.add_argument("--mamba-backend", type=str, default="auto", choices=["auto", "mamba_ssm", "mambapy"], help="Mamba backend (default: auto)")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap (default: 0.5)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # ---- convert ----
    c = sub.add_parser("convert", help="Convert checkpoint between pth and safetensors formats")
    c.add_argument("src", type=str, help="Source checkpoint path")
    c.add_argument("dst", type=str, help="Destination checkpoint path")
    c.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser


def _cmd_predict(args: argparse.Namespace) -> None:
    from imfuse_infer.predictor import IMFusePredictor

    input_paths: dict[str, str | None] = {
        "flair": args.flair,
        "t1ce": args.t1ce,
        "t1": args.t1,
        "t2": args.t2,
    }
    # Filter out None values
    input_paths = {k: v for k, v in input_paths.items() if v is not None}

    if not input_paths:
        print("Error: at least one modality must be provided (--flair, --t1ce, --t1, --t2)", file=sys.stderr)
        sys.exit(1)

    # Validate input files exist
    for name, path in input_paths.items():
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = IMFusePredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        num_cls=args.num_cls,
        interleaved_tokenization=not args.no_interleaved,
        mamba_skip=not args.no_mamba_skip,
        mamba_backend=args.mamba_backend,
        overlap=args.overlap,
    )

    seg = predictor.predict_nifti(input_paths, args.output)

    import numpy as np

    unique, counts = np.unique(seg, return_counts=True)
    label_info = ", ".join(f"{u}: {c}" for u, c in zip(unique, counts))
    print(f"Saved segmentation to {args.output}")
    print(f"  Shape: {seg.shape}, Labels: {{{label_info}}}")


def _cmd_convert(args: argparse.Namespace) -> None:
    from imfuse_infer.io.checkpoint import convert_to_safetensors, load_checkpoint

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"Error: source not found: {src}", file=sys.stderr)
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.suffix == ".safetensors":
        convert_to_safetensors(src, dst)
        print(f"Converted {src} → {dst}")
    elif dst.suffix in (".pth", ".pt"):
        import torch

        sd = load_checkpoint(src)
        torch.save(sd, str(dst))
        print(f"Converted {src} → {dst}")
    else:
        print(f"Error: unsupported output format: {dst.suffix}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if args.command == "predict":
        _cmd_predict(args)
    elif args.command == "convert":
        _cmd_convert(args)


if __name__ == "__main__":
    main()
