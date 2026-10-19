"""Test the Mnist class on your own handwritten digit images.

Usage
    python mnist_test.py                     # test every image in handwritten_digits/
    python mnist_test.py 2_1.png             # one image (true digit taken from the name)
    python mnist_test.py 2_1.png 2           # one image with the true digit given
    python mnist_test.py --show              # also display each image with imshow

Images are named n_m.png (n = digit, m = version), e.g. 0_1.png ... 9_5.png.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt

from mnist import Mnist

DIGIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handwritten_digits")


def resolve(path):
    """Accept either a full path or just a file name inside handwritten_digits/."""
    if os.path.exists(path):
        return path
    return os.path.join(DIGIT_DIR, path)


def sort_key(path):
    n, m = os.path.splitext(os.path.basename(path))[0].split("_")[:2]
    return int(n), int(m)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", help="image file (default: all in handwritten_digits/)")
    parser.add_argument("digit", nargs="?", type=int, help="true digit (default: from file name)")
    parser.add_argument("--show", action="store_true", help="display each image with imshow")
    args = parser.parse_args()

    mnist = Mnist()

    if args.image:
        files = [resolve(args.image)]
    else:
        files = sorted(glob.glob(os.path.join(DIGIT_DIR, "*_*.png")), key=sort_key)
        if not files:
            print(f"No images found in {DIGIT_DIR}. Create them first (see README.md).")
            return

    correct = 0
    for path in files:
        img, pred, true_digit, msg = mnist.test_image(path, args.digit)
        correct += pred == true_digit
        if args.show:
            plt.imshow(img, cmap="gray")
            plt.title(f"{os.path.basename(path)}  ->  predicted {pred}")
            plt.axis("off")
            plt.show()
        print(msg)

    if len(files) > 1:
        print(f"\nAccuracy: {correct}/{len(files)} = {correct / len(files):.1%}")


if __name__ == "__main__":
    main()
