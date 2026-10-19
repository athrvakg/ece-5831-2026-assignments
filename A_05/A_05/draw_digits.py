"""Draw your own handwritten digits with the mouse and save them as 28x28 PNGs.

    python draw_digits.py

The window asks for each digit in turn (0_1, 0_2, ... 9_5). Draw the digit,
press "Save & next" (or Enter). "Clear" (or Backspace) erases the canvas.
Images that already exist are skipped, so you can stop and resume any time.
Files are written to handwritten_digits/n_m.png (28 x 28, grayscale).
"""

import os
import tkinter as tk

from PIL import Image, ImageDraw

from mnist import Mnist

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "handwritten_digits")
SIZE = 280          # drawing area in pixels (10x the final 28x28)
PEN = 18            # pen width


class DigitPad:
    def __init__(self, root):
        self.root = root
        root.title("Handwritten digit pad")
        os.makedirs(OUT_DIR, exist_ok=True)
        self.todo = [(n, m) for n in range(10) for m in range(1, 6)
                     if not os.path.exists(os.path.join(OUT_DIR, f"{n}_{m}.png"))]

        self.label = tk.Label(root, font=("Arial", 16))
        self.label.pack(pady=6)
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="white", cursor="pencil")
        self.canvas.pack()
        bar = tk.Frame(root)
        bar.pack(pady=6)
        tk.Button(bar, text="Clear", width=10, command=self.clear).pack(side="left", padx=4)
        tk.Button(bar, text="Save & next", width=12, command=self.save).pack(side="left", padx=4)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start)
        root.bind("<Return>", lambda e: self.save())
        root.bind("<BackSpace>", lambda e: self.clear())

        self.last = None
        self.clear()
        self.update_label()

    def start(self, event):
        self.last = (event.x, event.y)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last:
            self.canvas.create_line(*self.last, x, y, width=PEN, capstyle="round", smooth=True)
            self.draw.line([self.last, (x, y)], fill=0, width=PEN)
            r = PEN // 2
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.last = (x, y)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (SIZE, SIZE), 255)   # white paper
        self.draw = ImageDraw.Draw(self.image)
        self.last = None

    def update_label(self):
        if self.todo:
            n, m = self.todo[0]
            self.label.config(text=f"Draw digit  {n}   (version {m} of 5)")
        else:
            self.label.config(text="All 50 digits saved - you can close this window.")

    def save(self):
        if not self.todo:
            return
        n, m = self.todo.pop(0)
        arr = Mnist.preprocess(self.image)
        Image.fromarray(arr).save(os.path.join(OUT_DIR, f"{n}_{m}.png"))
        print(f"saved {n}_{m}.png")
        self.clear()
        self.update_label()


if __name__ == "__main__":
    root = tk.Tk()
    DigitPad(root)
    root.mainloop()
