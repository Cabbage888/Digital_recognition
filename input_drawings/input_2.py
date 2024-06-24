import tkinter as tk

import matplotlib
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torchvision.__version__)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.lrelu1 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.lrelu2 = torch.nn.LeakyReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.lrelu3 = torch.nn.LeakyReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(256 * 6 * 6, 1024)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.lrelu4 = torch.nn.LeakyReLU()

        self.fc2 = torch.nn.Linear(1024, 256)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.lrelu5 = torch.nn.LeakyReLU()

        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.lrelu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.lrelu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.lrelu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.dropout1(self.lrelu4(self.fc1(x)))
        x = self.dropout2(self.lrelu5(self.fc2(x)))

        x = self.fc3(x)
        return x


# Load the trained model
net = Net()
# net.load_state_dict(torch.load('../model/model_2_1.pth'))
# net.load_state_dict(torch.load('D:\py\Digital_recognition\model\model_2_2.pth'))
net.load_state_dict(torch.load('D:\py\Digital_recognition\create_my_data\model_optimized_with_custom_data_5.pth'))
net.eval()  # eval模式是模型在预测的时候，会自动进行BN和Dropout的操作 训练的时候是不需要的 标志从训练模式切换到评估或预测模式
# BN的具体操作包括：
# 计算每个小批量数据（batch）的均值和方差。
# 使用这些统计量对小批量数据进行标准化，即 ((x - \mu) / \sqrt{\sigma^2 + \epsilon})，其中 (\mu) 是均值，(\sigma^2) 是方差，(\epsilon) 是一个很小的常数，用于避免除以零的情况。
# 可选地，还包含两个可学习的参数（缩放因子 (\gamma) 和偏移 (\beta)），用于恢复网络的表达能力：(\gamma(x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta)。
# 在.eval()模式下，对于BN层：
# 不再使用当前小批量数据计算均值和方差，而是使用在训练过程中积累的统计量（通常是整个训练集的移动平均）。这样做确保了预测时的输出一致性，不受小批量数据随机性的影响。
# Dropout层在评估时通常会被关闭（或设置为不丢弃任何单元），以利用模型的全部容量进行预测。

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the canvas size and the brush size
CANVAS_SIZE = 280
BRUSH_SIZE = 5


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

        self.label = tk.Label(root, text="", font=("Helvetica", 24))
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
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

    # def collect_data(self, image=None):  # 备用
    #
    #     return image

    def predict_digit(self, image=None):
        """
        预测当前画布上绘制的数字，并在数字未处于画布中心时才使用ROI进行预测。

        方法首先检查轮廓的中心是否偏离画布中心一定阈值，如果是，则仅对ROI进行处理和预测。
        否则，假设数字大致居中，直接使用整个画布进行预测。
        """
        if image is None:
            # 从画布中获取当前的图像
            image = self.image
        # 从画布中获取当前的图像
        # image = self.image
        # image = image.collect_data(image)
        # image = image_
        image_np = np.array(image)

        # 使用OpenCV寻找图像中的轮廓
        contours, _ = cv2.findContours(image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓
        contour_area = cv2.contourArea(contours[0])

        # 初始化变量以存储轮廓的质心
        centroid_x, centroid_y = 0, 0

        # 如果找到了轮廓
        if contours and len(contours) == 1:
            # 计算轮廓的质心
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

            # 定义画布中心点
            canvas_center_x, canvas_center_y = image.width // 2, image.height // 2

            # 设置偏离中心的阈值，例如画布宽度或高度的1/4
            threshold = min(image.width, image.height) // 6

            # 检查质心是否偏离中心超过阈值
            if abs(centroid_x - canvas_center_x) > threshold or abs(
                    centroid_y - canvas_center_y) > threshold or contour_area < 2100:
                # 提取轮廓的最小外接矩形
                x, y, w, h = cv2.boundingRect(contours[0])  # 获取轮廓的最小外接矩形
                # roi = image_np[y-20:y + h+20, x-20:x + w+20]

                # 扩展边界框的大小
                x -= 20
                y -= 25
                w += 40
                h += 45
                # 检查边界框是否超出画布范围
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x + w > CANVAS_SIZE:
                    w = CANVAS_SIZE - x
                if y + h > CANVAS_SIZE:
                    h = CANVAS_SIZE - y

                roi = image_np[y:y + h, x:x + w]  # 提取ROI

                #### 调试 ####
                # # 绘制ROI的边界框
                # color = (255, 0, 0)  # 设置颜色
                # thickness = 2  # 设置线条粗细
                # image_with_roi_box = cv2.rectangle(image_np, (x - 20, y - 20), (x + w + 20, y + h + 20), color,thickness)
                # # 显示或保存带有ROI框的图像
                # cv2.imshow("ROI Rectangle", image_with_roi_box)
                # cv2.waitKey(0)  # 等待按键后关闭窗口
                # cv2.destroyAllWindows()
                ###################

                roi_image = Image.fromarray(roi)  # 将ROI转换为PIL Image对象
                roi_image = roi_image.resize((28, 28)).convert('L')
            else:
                # 数字大致居中，直接使用整个画布
                roi_image = image.resize((28, 28)).convert('L')
        else:
            # 未找到轮廓，使用整个画布
            roi_image = image.resize((28, 28)).convert('L')

        image_tensor = transform(roi_image).unsqueeze(0)

        with torch.no_grad():
            output = net(image_tensor)
            prediction = torch.argmax(output, dim=1).item()

        self.label.config(text=f"Predicted Digit: {prediction}")
        print(f"Predicted Digit: {prediction}")
        self.show_image(roi_image)  # 使用QT输入时要注释掉
        return prediction

    def show_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.title("Input Image")
        plt.show()


if __name__ == "__main__":
    # 按图像路径读取图像
    # # image_path = input()  # 读入图像路径
    # image_path = sys.argv[1]  # 读取QT模拟的cmd指令
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # # cv2.imshow("image", image)
    # # cv2.waitKey(0)  # 等待按键操作，0表示无限等待
    # # cv2.destroyAllWindows()  # 关闭所有cv2创建的窗口
    #
    # # 检查图像是否读取成功
    # if image is None:
    #     print("图像读取失败，请检查路径是否正确")
    # else:
    #     # 判断图像通道数
    #     channels = image.shape[2] if len(image.shape) == 3 else 1
    #
    #     # 如果是三通道图像，则转换为单通道（灰度图）
    #     if channels == 3:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         # print("图像已转换为单通道。")
    #     else:
    #         image = image
    #
    # # 转换为PIL图像
    # image = Image.fromarray(image).convert('L')  # 确保图像以标准的灰度模式（L模式）
    #
    # root = tk.Tk()
    # app = App(root)
    # result = app.predict_digit(image)  # 识别出的数字
    # print(result)

    # 以下命令用来启动画布 用画布读取图像
    root = tk.Tk()
    app = App(root)
    root.mainloop()
