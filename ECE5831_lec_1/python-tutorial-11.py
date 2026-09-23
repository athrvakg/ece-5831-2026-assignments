# Section 11. Modules & Packages
# pythontutorial.net - Python Basics

# --- Modules ---
# A module is simply a .py file containing Python code (functions, classes, variables)
# that can be imported and reused. The standard library "math" is a built-in module.
import math
print("Modules:")
print("sqrt(16) =", math.sqrt(16))
print("pi =", math.pi)

# You can also import specific names from a module
from math import sqrt, pi
print(sqrt(25), pi)

# --- Module Search Path ---
# When you `import x`, Python searches sys.path (current dir, PYTHONPATH,
# installation-dependent default paths) in order to locate the module.
import sys
print("\nModule Search Path (first 3 entries):")
for p in sys.path[:3]:
    print(" -", p)

# --- __name__ variable ---
# Every module has a built-in __name__ variable. When a file is run directly,
# __name__ == "__main__". When it's imported, __name__ == the module's name.
print("\n__name__ variable:")
print("This file's __name__ is:", __name__)

def main():
    print("main() executed because this file was run directly, not imported.")

if __name__ == "__main__":
    main()

# --- Packages ---
# A package is a directory containing multiple modules plus an __init__.py file
# (in modern Python, __init__.py is optional but still commonly used) that lets
# you organize related modules into a namespace, e.g.:
#
#   mypackage/
#       __init__.py
#       module_a.py
#       module_b.py
#
# Usage would then look like:
#   import mypackage.module_a
#   from mypackage import module_b
print("\nPackages:")
print("A package is a folder of modules with (optionally) an __init__.py,")
print("imported like: import mypackage.module_a")

# --- Private Functions ---
# Python has no true "private" access modifier. By convention, a single leading
# underscore signals "internal use only" - it's not enforced, just a convention.
def _internal_helper():
    return "I'm intended for internal use only."

def public_function():
    return _internal_helper() + " (called from a public function)"

print("\nPrivate Functions (convention):")
print(public_function())
print(_internal_helper())  # still callable - Python doesn't truly restrict it

if __name__ == "__main__":
    print("\nSection 11 demo complete.")
