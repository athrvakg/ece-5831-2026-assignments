"""module3.py

Tests the LogicGate class from logic_gate.py by printing the truth table of
every gate and checking it against the expected output.
"""

import numpy as np

from logic_gate import LogicGate


def show(name, gate, expected):
    """Print a gate's truth table and assert it matches the expected outputs."""
    table = LogicGate.truth_table(gate)
    print(f"{name}")
    print(" a  b | out")
    print("-------------")
    for a, b, out in table:
        print(f" {a}  {b} |  {out}")
    print()
    assert np.array_equal(table[:, 2], expected), f"{name} produced a wrong truth table"


def main():
    gate = LogicGate()

    # Expected outputs for inputs (0,0), (0,1), (1,0), (1,1)
    show("AND gate", gate.and_gate, [0, 0, 0, 1])
    show("NAND gate", gate.nand_gate, [1, 1, 1, 0])
    show("OR gate", gate.or_gate, [0, 1, 1, 1])
    show("NOR gate", gate.nor_gate, [1, 0, 0, 0])
    show("XOR gate", gate.xor_gate, [0, 1, 1, 0])

    # Element-wise use on NumPy arrays
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    print("Array inputs  a =", a, " b =", b)
    print("and_gate :", gate.and_gate(a, b))
    print("nand_gate:", gate.nand_gate(a, b))
    print("or_gate  :", gate.or_gate(a, b))
    print("nor_gate :", gate.nor_gate(a, b))
    print("xor_gate :", gate.xor_gate(a, b))

    # Invalid input handling
    try:
        gate.and_gate(2, 1)
    except ValueError as err:
        print("\nInvalid input correctly rejected:", err)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
