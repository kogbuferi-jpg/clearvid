"""ClearVid CLI entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.pipeline import remove_subtitles


def main():
    p = argparse.ArgumentParser(
        prog="clearvid",
        description="Remove hardcoded subtitles / watermarks from videos locally.")
    p.add_argument("--input", required=True, help="Input video path (or folder, todo)")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--engine", choices=["e2fgvi", "propainter"], default=None,
                   help="Inpainting engine (default: from config)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--keep-temp", action="store_true",
                   help="Keep intermediate frames/masks for debugging")
    args = p.parse_args()

    if not Path(args.input).exists():
        p.error(f"input not found: {args.input}")
    remove_subtitles(args.input, args.output, config_path=args.config,
                     engine_override=args.engine, keep_temp=args.keep_temp)


if __name__ == "__main__":
    main()
