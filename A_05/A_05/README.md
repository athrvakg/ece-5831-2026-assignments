# Forward Propagation Implementation

ECE 5831 – Pattern Recognition & Neural Networks · Atharva Gujarathi

Two inference (forward-propagation) tasks:

1. **Rock–Paper–Scissors** – an image classifier trained with Google Teachable Machine, loaded in Python and used to classify new images.
2. **MNIST** – a `Mnist` class implementing a pretrained 3-layer network (784 → 50 → 100 → 10) from scratch with NumPy, tested on my own handwritten digits.

## Files

```
A_05/
├── rock-paper-scissors.ipynb   # Problem 1 notebook: load model, take new image, forward pass, classify
├── rock_paper_scissors.py      # RockPaperScissors class (+ command-line use)
├── model/
│   ├── keras_model.h5          # trained Teachable Machine model (Keras export)
│   └── labels.txt              # 0 Rock / 1 Paper / 2 Scissors
├── rps_images/                 # new test images (not used in training)
│
├── mnist.ipynb                 # Problem 2 notebook: imshow each digit + Success/Fail result
├── mnist.py                    # Mnist class (sigmoid/softmax forward propagation)
├── sample_weight.pkl           # pretrained weights W1,b1,W2,b2,W3,b3
├── mnist_test.py               # command-line test: python mnist_test.py [n_m.png] [digit]
├── handwritten_digits/         # my handwriting: 0_1.png … 9_5.png (28 x 28 x 1 grayscale)
├── draw_digits.py              # helper: draw digits with the mouse and save 28x28 PNGs
├── make_digit_images.py        # helper: convert photos of handwriting into 28x28 PNGs
├── requirements.txt
└── README.md
```

## Setup

```bash
conda activate ece-5831-2026
pip install -r requirements.txt
```

`tf-keras` is needed because Teachable Machine exports a Keras 2 `.h5` model, which the Keras 3 bundled with TensorFlow ≥ 2.16 cannot read reliably.

## Problem 1: Rock–Paper–Scissors

**Training (Teachable Machine).** Image Project → Standard image model → three classes `Rock`, `Paper`, `Scissors`, each filled with webcam samples of my hand (different angles, distances and backgrounds) → *Train Model* → *Export Model* → *Tensorflow* → *Keras* → *Download my model*. The unzipped `keras_model.h5` and `labels.txt` go in `model/`.

**Inference.** `RockPaperScissors` loads the model and labels, accepts a new image, and:

1. center-crops/resizes it to 224 × 224 RGB and scales pixels to [-1, 1] (same as Teachable Machine);
2. runs forward propagation through the network (MobileNet feature extractor → dense layers → softmax);
3. returns the class with the highest probability.

The notebook also performs the forward pass block by block and checks it matches the full model output.

```bash
python rock_paper_scissors.py rps_images/my_rock.jpg
```

## Problem 2: MNIST

`Mnist` (in `mnist.py`) loads `sample_weight.pkl` and computes

```
z1 = sigmoid(x · W1 + b1)      # 784 -> 50
z2 = sigmoid(z1 · W2 + b2)     # 50  -> 100
y  = softmax(z2 · W3 + b3)     # 100 -> 10
```

It scores 93.5 % on the 10,000-image MNIST test set.

**My handwriting.** Five images per digit, named `n_m.png` (e.g. `0_1.png` = digit 0, version 1), 28 × 28 × 1 grayscale, in `handwritten_digits/`. They were created with one of the helpers:

- `python draw_digits.py` – draw each digit with the mouse; or
- `python make_digit_images.py --sheet sheet.jpg` – photo of a 10 × 5 grid written on paper (row = digit, column = version).

Both apply MNIST-style normalisation (white digit on black, fitted to a 20 × 20 box and centred by centre of mass in 28 × 28).

**Testing.** `mnist.ipynb` shows every image with `imshow` and prints, for example:

```
Success: Image 2_1.png is for digit 2 is recognized as 2.
Fail: Image 2_1.png is for digit 2 but the inference result is 3.
```

followed by overall accuracy, per-digit results and a confusion matrix. From the terminal:

```bash
python mnist_test.py              # all 50 images
python mnist_test.py 2_1.png 2    # one image
```
