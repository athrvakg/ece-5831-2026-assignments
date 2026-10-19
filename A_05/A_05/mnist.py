"""Mnist: a 3-layer neural network that recognizes handwritten digits.

The network uses pretrained weights (``sample_weight.pkl``) and only performs
forward propagation (inference):

    input (784) -> [W1, b1] -> sigmoid -> hidden 1 (50)
                -> [W2, b2] -> sigmoid -> hidden 2 (100)
                -> [W3, b3] -> softmax -> output   (10 class probabilities)

Images must be 28 x 28 grayscale. MNIST digits are white strokes on a black
background with values scaled to [0, 1], so ``load_image`` converts a normal
"dark ink on white paper" image into that format.
"""

import os
import pickle

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))


class Mnist:
    """Forward-propagation MNIST classifier using pretrained weights."""

    def __init__(self, weight_file=None):
        if weight_file is None:
            weight_file = os.path.join(_HERE, "sample_weight.pkl")
        self.weight_file = weight_file
        self.params = {}
        self.init_network()

    # ------------------------------------------------------------------
    # Network setup
    # ------------------------------------------------------------------
    def init_network(self):
        """Load the pretrained weights W1, b1, W2, b2, W3, b3."""
        with open(self.weight_file, "rb") as f:
            self.params = pickle.load(f)
        return self.params

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def softmax(a):
        """Numerically stable softmax (works on one sample or a batch)."""
        a = np.asarray(a, dtype=np.float64)
        a = a - np.max(a, axis=-1, keepdims=True)  # avoid overflow
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, x):
        """Forward propagation.

        x : array of shape (784,) for one image or (N, 784) for a batch,
            values in [0, 1].
        Returns the softmax output: shape (10,) or (N, 10).
        """
        W1, W2, W3 = self.params["W1"], self.params["W2"], self.params["W3"]
        b1, b2, b3 = self.params["b1"], self.params["b2"], self.params["b3"]

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)
        return y

    # Same thing, under the name used in the textbook.
    predict = forward

    def predict_digit(self, x):
        """Return (predicted digit, softmax probabilities)."""
        y = self.forward(x)
        return int(np.argmax(y, axis=-1)), y

    # ------------------------------------------------------------------
    # Image loading / pre-processing
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess(img, size=28, box=20):
        """Turn any grayscale digit image into an MNIST-style 28 x 28 image.

        Steps (the same idea used to build the original MNIST dataset):
          1. make the digit bright on a dark background (invert if needed)
          2. remove background noise and stretch contrast to 0..255
          3. crop to the digit's bounding box
          4. resize so the longer side is ``box`` (20) pixels, keeping aspect
          5. paste into a 28 x 28 canvas centred on the centre of mass
        Returns a uint8 array of shape (28, 28).
        """
        a = np.asarray(img.convert("L"), dtype=np.float64)

        # 1. MNIST digits are white on black. Paper photos are dark on white.
        if np.median(a) > 127:
            a = 255.0 - a

        # 2. remove the background level and stretch contrast
        bg = np.percentile(a, 50)
        a = np.clip(a - bg, 0, None)
        if a.max() > 0:
            a = a / a.max() * 255.0
        a[a < 40] = 0  # kill faint noise / paper texture

        # 3. crop to bounding box of the strokes
        ys, xs = np.nonzero(a)
        if len(xs) == 0:  # blank image
            return np.zeros((size, size), dtype=np.uint8)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        # 4. fit into a box x box square, keeping the aspect ratio
        h, w = a.shape
        scale = box / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        digit = Image.fromarray(a.astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
        d = np.asarray(digit, dtype=np.float64)

        # 5. centre by centre of mass in a size x size canvas
        canvas = np.zeros((size, size), dtype=np.float64)
        total = d.sum()
        if total > 0:
            cy = (d.sum(axis=1) * np.arange(nh)).sum() / total
            cx = (d.sum(axis=0) * np.arange(nw)).sum() / total
        else:
            cy, cx = nh / 2, nw / 2
        top = int(round(size / 2 - cy))
        left = int(round(size / 2 - cx))
        top = min(max(top, 0), size - nh)
        left = min(max(left, 0), size - nw)
        canvas[top:top + nh, left:left + nw] = d
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def load_image(self, path, preprocess=True):
        """Read an image file and return (image_28x28_uint8, input_vector_784).

        If the file is already an MNIST-style 28 x 28 image (white digit on a
        black background) set ``preprocess=False`` to use it as-is.
        """
        img = Image.open(path)
        if preprocess:
            arr = self.preprocess(img)
        else:
            arr = np.asarray(img.convert("L").resize((28, 28)), dtype=np.uint8)
        x = arr.reshape(784).astype(np.float32) / 255.0  # normalize to [0, 1]
        return arr, x

    # ------------------------------------------------------------------
    # Convenience: classify a file and build the required message
    # ------------------------------------------------------------------
    def test_image(self, path, true_digit=None):
        """Classify one image file.

        ``true_digit`` defaults to the first part of the file name
        (``n_m.png`` -> n). Returns (image, predicted, true_digit, message).
        """
        name = os.path.basename(path)
        if true_digit is None:
            true_digit = int(name.split("_")[0])
        img, x = self.load_image(path)
        pred, _ = self.predict_digit(x)
        if pred == true_digit:
            msg = f"Success: Image {name} is for digit {true_digit} is recognized as {pred}."
        else:
            msg = (f"Fail: Image {name} is for digit {true_digit} but the "
                   f"inference result is {pred}.")
        return img, pred, true_digit, msg
