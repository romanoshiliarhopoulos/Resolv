import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from . import pipeline

load_dotenv()

TYPES = ["hdris", "models", "textures", "ambientcg", "all"]


def main():
    sketchfab_token = os.getenv("SKETCHFAB_API_TOKEN")
    parser = argparse.ArgumentParser(description="Download assets for Resolv dataset generation")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Root output directory")
    parser.add_argument("--sketchfab-token", type=str, default=sketchfab_token, help="Sketchfab API token")
    parser.add_argument("--limit", type=int, default=None, help="Max assets per source (for testing)")
    parser.add_argument(
        "--type", dest="asset_type", choices=TYPES, default="all",
        help="Which asset type to download: hdris, models, textures (Polyhaven), ambientcg, or all",
    )
    args = parser.parse_args()

    pipeline.run(
        output_dir=args.output,
        sketchfab_token=args.sketchfab_token,
        limit=args.limit,
        asset_type=args.asset_type,
    )


if __name__ == "__main__":
    main()