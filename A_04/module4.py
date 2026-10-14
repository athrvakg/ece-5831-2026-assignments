"""module4.py - imports and tests the MultilayerPerceptron class."""

import numpy as np

from multilayer_perceptron import MultilayerPerceptron


def reference_forward(net, x):
    """Independent re-computation using plain math (no class methods)."""
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    a = np.asarray(x, dtype=float)
    for i in (1, 2, 3):
        a = sig(a @ net[f"W{i}"] + net[f"b{i}"])
    return np.maximum(0.0, a @ net["W4"] + net["b4"])


def main():
    mlp = MultilayerPerceptron()

    print("Layer sizes: 3 -> 5 -> 5 -> 2 -> 2")
    for name in ("W1", "W2", "W3", "W4"):
        print(f"{name} shape: {mlp.network[name].shape}")
    print()

    inputs = [
        np.array([1.0, 0.5, -1.5]),
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, -1.0, 3.0]),
        np.array([-1.0, 2.0, 0.5]),
    ]

    for x in inputs:
        y, layers = mlp.forward(x, return_all=True)
        expected = reference_forward(mlp.network, x)
        print(f"input  : {x}")
        print(f"  layer 1 (sigmoid): {np.round(layers[1], 4)}")
        print(f"  layer 2 (sigmoid): {np.round(layers[2], 4)}")
        print(f"  layer 3 (sigmoid): {np.round(layers[3], 4)}")
        print(f"  output  (ReLU)   : {np.round(y, 4)}")
        assert y.shape == (2,)
        assert np.allclose(y, expected), "mismatch with reference computation"
        assert np.all(y >= 0), "ReLU output must be non-negative"
        for k in (1, 2, 3):
            assert np.all((layers[k] > 0) & (layers[k] < 1)), "sigmoid range violated"
        print()

    # Batch input: all rows at once should match one-at-a-time results.
    batch = np.vstack(inputs)
    batch_out = mlp.forward(batch)
    single_out = np.vstack([mlp.forward(x) for x in inputs])
    assert batch_out.shape == (len(inputs), 2)
    assert np.allclose(batch_out, single_out)
    print("batch output:\n", np.round(batch_out, 4))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
