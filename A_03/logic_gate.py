"""logic_gate.py

A small library of logic gates implemented with NumPy.

Every gate works on scalars as well as on array-like inputs (lists, tuples,
NumPy arrays) and is applied element-wise.  Inputs must contain only 0/1
(or True/False) values.  Results are returned as integers (0 or 1).
"""

import numpy as np


class LogicGate:
    """Collection of basic logic gates built on top of NumPy.

    All gates are static methods, so they can be used either directly from
    the class (``LogicGate.and_gate(1, 0)``) or from an instance
    (``LogicGate().and_gate(1, 0)``).
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_binary(x):
        """Validate that ``x`` only holds 0/1 values and return it as bool array."""
        arr = np.asarray(x)
        if not np.all(np.isin(arr, [0, 1])):
            raise ValueError("Logic gate inputs must contain only 0/1 or True/False values.")
        return arr.astype(bool)

    @staticmethod
    def _to_int(result):
        """Convert a boolean result back to integer 0/1."""
        return np.asarray(result).astype(int)

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    @staticmethod
    def and_gate(a, b):
        """AND gate: output is 1 only when both inputs are 1."""
        a, b = LogicGate._to_binary(a), LogicGate._to_binary(b)
        return LogicGate._to_int(np.logical_and(a, b))

    @staticmethod
    def nand_gate(a, b):
        """NAND gate: the negation of AND."""
        a, b = LogicGate._to_binary(a), LogicGate._to_binary(b)
        return LogicGate._to_int(np.logical_not(np.logical_and(a, b)))

    @staticmethod
    def or_gate(a, b):
        """OR gate: output is 1 when at least one input is 1."""
        a, b = LogicGate._to_binary(a), LogicGate._to_binary(b)
        return LogicGate._to_int(np.logical_or(a, b))

    @staticmethod
    def nor_gate(a, b):
        """NOR gate: the negation of OR."""
        a, b = LogicGate._to_binary(a), LogicGate._to_binary(b)
        return LogicGate._to_int(np.logical_not(np.logical_or(a, b)))

    @staticmethod
    def xor_gate(a, b):
        """XOR gate: output is 1 when exactly one input is 1."""
        a, b = LogicGate._to_binary(a), LogicGate._to_binary(b)
        return LogicGate._to_int(np.logical_xor(a, b))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def truth_table(gate):
        """Return the truth table of a two-input gate as a (4, 3) NumPy array.

        Columns are ``[a, b, output]``.
        """
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = gate(inputs[:, 0], inputs[:, 1])
        return np.column_stack((inputs, outputs))
