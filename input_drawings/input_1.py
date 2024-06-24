import tkinter as tk
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'GTK3Agg' 等
import matplotlib.pyplot as plt


# Define the Net class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output


# Load the trained model
net = torch.load('../model/model_1.pth')
net.eval()  # 模型在评估模式下，会自动关闭dropout和batch normalization

# 定义变换 将图片resize成28x28，然后转换为tensor，最后归一化 因为神经网络输入是28x28的灰度图像，所以这里需要resize
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 均值为0.5，方差为0.5
])

# Define the canvas size and the brush size
CANVAS_SIZE = 280
BRUSH_SIZE = 10


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a digit")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)

        self.label = tk.Label(root, text="", font=("Helvetica", 24)) # 创建一个标签控件，用于显示预测结果
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - BRUSH_SIZE), (event.y - BRUSH_SIZE)
        x2, y2 = (event.x + BRUSH_SIZE), (event.y + BRUSH_SIZE)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="black")
        self.label.config(text="")

    def predict_digit(self):
        image = self.image.resize((28, 28))
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = net(image)
            prediction = torch.argmax(output, dim=1).item()

        self.label.config(text=f"Predicted Digit: {prediction}")
        print(f"Predicted Digit: {prediction}")
        self.show_image(image)

    def show_image(self, image):
        plt.imshow(image.squeeze())
        plt.title("Input Image")
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()  # 启动主循环
    # plt.show() # 显示图像
    print("Done")
