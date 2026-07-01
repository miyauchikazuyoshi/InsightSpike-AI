#!/usr/bin/env python3
"""
論文PDF (C000993.pdf) から図3 (迷路) と図4 (BERT箱ひげ) を抽出する。

元のPDFからの再生成用スクリプト。PDF の位置が変わった場合はトリミング座標を
調整する必要がある。

依存:
  - pdftoppm (poppler-utils)
  - Pillow (pip install Pillow)

使い方:
  python3 src/extract_figures_from_pdf.py path/to/C000993.pdf
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "figures"

# 200dpi レンダリング時の座標 (左上x, 左上y, 右下x, 右下y)
# 論文 page 3 上での図の位置
CROP_BOXES = {
    "fig3_maze.png": (150, 705, 830, 1100),
    "fig4_bert.png": (830, 175, 1640, 515),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path, help="C000993.pdf へのパス")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--page", type=int, default=3, help="図が含まれるページ番号 (1-indexed)")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        raise SystemExit(f"PDF not found: {args.pdf_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        prefix = td_path / "page"
        subprocess.run(
            ["pdftoppm", "-r", str(args.dpi), "-png",
             "-f", str(args.page), "-l", str(args.page),
             str(args.pdf_path), str(prefix)],
            check=True,
        )

        # pdftoppm output pattern: page-<N>.png
        page_images = sorted(td_path.glob("page-*.png"))
        if not page_images:
            raise SystemExit("pdftoppm produced no output")
        page_img = Image.open(page_images[0])
        print(f"Rendered page size: {page_img.size}")

        for filename, box in CROP_BOXES.items():
            cropped = page_img.crop(box)
            out_path = OUT_DIR / filename
            cropped.save(out_path)
            print(f"  [OK] {filename} {cropped.size} -> {out_path}")


if __name__ == "__main__":
    main()
