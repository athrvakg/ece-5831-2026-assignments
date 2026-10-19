"""Rock-Paper-Scissors classifier using a Teachable Machine image model.

Export your model from Teachable Machine (Export Model -> Tensorflow -> Keras ->
Download my model). Unzip it and put both files in the model/ folder:

    model/keras_model.h5
    model/labels.txt

Usage
    python rock_paper_scissors.py rps_images/my_hand.jpg
"""

import os
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "keras_model.h5")
LABELS_PATH = os.path.join(HERE, "model", "labels.txt")
IMG_SIZE = 224  # Teachable Machine image models take 224 x 224 x 3 input


def _keras():
    """Return a Keras 2 compatible API.

    Teachable Machine exports a Keras 2 .h5 file. With TensorFlow >= 2.16 the
    built-in Keras is Keras 3, which often cannot read it, so we prefer the
    ``tf_keras`` package (pip install tf-keras) and fall back to tf.keras.
    """
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras
    return keras


def load_teachable_machine_model(model_path=MODEL_PATH):
    keras = _keras()

    # Newer Teachable Machine exports save a `groups` argument for
    # DepthwiseConv2D that Keras does not accept -> drop it when loading.
    class DepthwiseConv2DCompat(keras.layers.DepthwiseConv2D):
        def __init__(self, *args, groups=None, **kwargs):
            super().__init__(*args, **kwargs)

    try:
        return keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"DepthwiseConv2D": DepthwiseConv2DCompat},
        )
    except Exception as err:
        raise RuntimeError(
            f"Could not load {model_path}.\n"
            "Teachable Machine models need the Keras 2 API. Install it into the "
            "same environment as TensorFlow and retry:\n"
            "    pip install tf-keras\n"
            f"(original error: {type(err).__name__}: {err})"
        ) from err


def load_labels(labels_path=LABELS_PATH):
    """labels.txt lines look like '0 Rock' -> return ['Rock', ...]."""
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [ln.split(" ", 1)[1] if " " in ln else ln for ln in lines]


class RockPaperScissors:
    """Loads the trained model and classifies a new image."""

    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model = load_teachable_machine_model(model_path)
        self.labels = load_labels(labels_path)

    @staticmethod
    def preprocess(image):
        """Same pre-processing Teachable Machine uses during training.

        1. RGB, center-crop and resize to 224 x 224
        2. scale pixels from [0, 255] to [-1, 1]
        3. add a batch dimension -> shape (1, 224, 224, 3)
        """
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32)
        normalized = (arr / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0), image

    def forward(self, x):
        """Forward propagation (inference) through the trained network."""
        return self.model(x, training=False).numpy()

    def forward_layer_by_layer(self, x, verbose=True):
        """Forward propagation done explicitly, one layer (block) at a time.

        Gives the same answer as ``forward`` but shows how the input tensor is
        transformed as it passes through the network.
        """
        def blocks(model):
            # Teachable Machine nests Sequential models; unroll them. A
            # Functional sub-model (MobileNet, which has skip connections)
            # is applied as one block.
            for layer in model.layers:
                if layer.__class__.__name__ == "Sequential":
                    yield from blocks(layer)
                else:
                    yield layer

        a = x
        for layer in blocks(self.model):
            a = layer(a, training=False)
            if verbose:
                print(f"{layer.name:28s} {layer.__class__.__name__:24s} "
                      f"-> output shape {tuple(a.shape)}")
        return np.asarray(a)

    def predict(self, image):
        """Classify an image (path or PIL image).

        Returns (label, confidence, probabilities, resized_image).
        """
        x, resized = self.preprocess(image)
        probs = self.forward(x)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx]), probs, resized


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rps = RockPaperScissors()
    for path in sys.argv[1:]:
        label, conf, probs, _ = rps.predict(path)
        detail = ", ".join(f"{l}: {p:.3f}" for l, p in zip(rps.labels, probs))
        print(f"{os.path.basename(path)} -> {label} ({conf:.1%})   [{detail}]")


if __name__ == "__main__":
    main()
