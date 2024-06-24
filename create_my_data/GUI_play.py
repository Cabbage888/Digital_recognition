# 可变颜色画布，笔刷 尚在开发中 2023.05.09
import tkinter as tk
from tkinter import colorchooser, messagebox
from PIL import Image, ImageDraw, ImageTk
import torch
import torchvision.transforms as transforms
import os


# Load the trained model
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.model = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 16, 3, 1, 1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2, 2),
#             torch.nn.Conv2d(16, 32, 3, 1, 1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2, 2),
#             torch.nn.Conv2d(32, 64, 3, 1, 1),
#             torch.nn.ReLU(),
#             torch.nn.Flatten(),
#             torch.nn.Linear(7 * 7 * 64, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 10),
#             torch.nn.Softmax(dim=1)
#         )
#
#     def forward(self, input):
#         output = self.model(input)
#         return output

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(128 * 3 * 3, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = Net().to(device)
net.load_state_dict(torch.load('./model_optimized_with_custom_data_3.pth', map_location=device))
net.eval()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Directory to save prediction results
results_dir = './results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# GUI Application
class DigitPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Predictor")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10, columnspan=4)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.brush_color = 'black'

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=1)

        self.canvas_color_button = tk.Button(root, text="Canvas Color", command=self.choose_canvas_color)
        self.canvas_color_button.grid(row=1, column=2)

        self.brush_color_button = tk.Button(root, text="Brush Color", command=self.choose_brush_color)
        self.brush_color_button.grid(row=1, column=3)

        self.results_frame = tk.Frame(root)
        self.results_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

        self.results_listbox = tk.Listbox(self.results_frame, width=50)
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH)

        self.results_scrollbar = tk.Scrollbar(self.results_frame)
        self.results_scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.results_listbox.config(yscrollcommand=self.results_scrollbar.set)
        self.results_scrollbar.config(command=self.results_listbox.yview)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.brush_color, width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill=self.brush_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")

    def choose_canvas_color(self):
        color_code = colorchooser.askcolor(title="Choose canvas color")
        if color_code:
            self.canvas.config(bg=color_code[1])
            self.image = Image.new("L", (280, 280), int(color_code[0][0]))
            self.draw = ImageDraw.Draw(self.image)

    def choose_brush_color(self):
        color_code = colorchooser.askcolor(title="Choose brush color")
        if color_code:
            self.brush_color = color_code[1]

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = net(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        img_filename = f"{results_dir}/{len(os.listdir(results_dir))}.png"
        img.save(img_filename)

        self.results_listbox.insert(tk.END, f"Prediction: {prediction}")
        self.results_listbox.see(tk.END)
        self.results_listbox.yview(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitPredictorApp(root)
    root.mainloop()
