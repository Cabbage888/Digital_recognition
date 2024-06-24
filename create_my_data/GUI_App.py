import tkinter as tk
from PIL import Image, ImageDraw
import os
import random

# Directories to save custom handwritten digits
train_custom_path = './custom_digits/train4/'
value_custom_path = './custom_digits/value4/'

# Ensure directories exist
for path in [train_custom_path, value_custom_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Define canvas size and brush size
CANVAS_SIZE = 280
BRUSH_SIZE = 5

class CollectDigitsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Collect Handwritten Digits")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        # Ensure the image is black (0) for a blank canvas
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)  # Use 0 for black background
        self.draw = ImageDraw.Draw(self.image)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_digit)
        self.save_button.pack(side=tk.LEFT)

        self.label = tk.Label(root, text="Enter digit and click Save", font=("Helvetica", 12))
        self.label.pack()

        self.digit_entry = tk.Entry(root, width=5)
        self.digit_entry.pack()

    def paint(self, event):
        x1, y1 = (event.x - BRUSH_SIZE), (event.y - BRUSH_SIZE)
        x2, y2 = (event.x + BRUSH_SIZE), (event.y + BRUSH_SIZE)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="black")  # Ensure clear color is black
        self.label.config(text="Enter digit and click Save")

    def save_digit(self):
        digit = self.digit_entry.get()
        # save_path = train_custom_path
        save_path = value_custom_path

        def get_unique_index_per_digit(path, digit):
            digit_files = [f for f in os.listdir(path) if f.startswith(f"{digit}_") and f.endswith('.png')]
            if digit_files:
                last_file = max(digit_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                return int(last_file.split('_')[1].split('.')[0]) + 1
            return 0

        if digit.isdigit() and 0 <= int(digit) <= 9:
            unique_index = get_unique_index_per_digit(save_path, digit)
            filename = f'{save_path}/{digit}_{unique_index}.png'

            self.image.resize((28, 28)).save(filename)
            self.label.config(text="Digit saved!")
        else:
            self.label.config(text="Invalid digit! Please enter a digit between 0 and 9.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CollectDigitsApp(root)
    root.mainloop()

