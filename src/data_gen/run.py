import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from . import pipeline

load_dotenv()


def main():
    sketchfab_token = os.getenv("SKETCHFAB_API_TOKEN")
    parser = argparse.ArgumentParser(description="Download assets for Resolv dataset generation")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Root output directory")
    parser.add_argument("--sketchfab-token", type=str, default=sketchfab_token, help="Sketchfab API token")
    parser.add_argument("--limit", type=int, default=None, help="Max assets per source (for testing)")
    args = parser.parse_args()

    pipeline.run(
        output_dir=args.output,
        sketchfab_token=args.sketchfab_token,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()