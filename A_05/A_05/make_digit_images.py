"""Convert photos/scans of YOUR handwriting into 28 x 28 grayscale MNIST images.

Output: handwritten_digits/n_m.png  (n = digit 0-9, m = version 1-5), 28x28, mode "L".

Option A - one photo of a sheet (easiest)
    On white paper write the digits in a 10 x 5 grid with a dark pen/marker:
        row 1: 0 0 0 0 0
        row 2: 1 1 1 1 1
        ...
        row 10: 9 9 9 9 9
    Keep the digits roughly evenly spaced, take a straight photo, crop it so only
    the grid is visible, then run:
        python make_digit_images.py --sheet my_sheet.jpg

Option B - one photo per digit
    Put photos named like 0_1.jpg, 0_2.jpg, ... 9_5.jpg in a folder, then run:
        python make_digit_images.py --folder my_photos

Either way a preview (digits_preview.png) is written so you can check the result.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageOps

from mnist import Mnist

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "handwritten_digits")


def save_digit(img, digit, version):
    arr = Mnist.preprocess(img)
    path = os.path.join(OUT_DIR, f"{digit}_{version}.png")
    Image.fromarray(arr).save(path)
    return path


def from_sheet(sheet_path, rows=10, cols=5, margin=0.12):
    """Split a cropped photo of a rows x cols grid into individual digits."""
    sheet = ImageOps.exif_transpose(Image.open(sheet_path)).convert("L")
    w, h = sheet.size
    cw, ch = w / cols, h / rows
    paths = []
    for r in range(rows):          # row r holds digit r
        for c in range(cols):      # column c is version c + 1
            box = (int((c + margin) * cw), int((r + margin) * ch),
                   int((c + 1 - margin) * cw), int((r + 1 - margin) * ch))
            paths.append(save_digit(sheet.crop(box), r, c + 1))
    return paths


def from_folder(folder):
    paths = []
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        parts = stem.split("_")
        if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        img = ImageOps.exif_transpose(Image.open(f)).convert("L")
        paths.append(save_digit(img, int(parts[0]), int(parts[1])))
    return paths


def preview(paths, out="digits_preview.png", scale=3):
    """Tile all generated 28x28 images (one row per digit) for a quick check."""
    by_name = {os.path.basename(p): p for p in paths}
    tile = 28 * scale
    canvas = Image.new("L", (5 * tile, 10 * tile), 0)
    for n in range(10):
        for m in range(1, 6):
            p = by_name.get(f"{n}_{m}.png")
            if p:
                im = Image.open(p).resize((tile, tile), Image.NEAREST)
                canvas.paste(im, ((m - 1) * tile, n * tile))
    canvas.save(os.path.join(HERE, out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sheet", help="photo of a 10 x 5 grid of digits (cropped to the grid)")
    g.add_argument("--folder", help="folder of photos named n_m.jpg / n_m.png")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = from_sheet(args.sheet) if args.sheet else from_folder(args.folder)
    print(f"Saved {len(paths)} images to {OUT_DIR}")
    print(f"Preview written to {preview(paths)}")


if __name__ == "__main__":
    main()
