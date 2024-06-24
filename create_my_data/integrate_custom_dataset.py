# 使用自定义数据集和MNIST的数据集进行训练
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'GTK3Agg' 等
import matplotlib.pyplot as plt

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_custom_path = './custom_digits/train4/'
value_custom_path = './custom_digits/value4/'

# 数据集路径
path = '../data/'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_images(images, labels, classes, num_images=50):
    rows = int(num_images ** 0.5)  # 计算行数
    cols = (num_images // rows) + (num_images % rows > 0)  # 计算列数
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # 创建一个画布
    for i in range(num_images):
        ax = axes[i // cols, i % cols]  # 获取当前子图
        img = images[i].permute(1, 2, 0).numpy()  # 将图像从 (C, H, W) 转换为 (H, W, C)
        img = (img * 0.5 + 0.5)  # 反归一化
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(classes[labels[i].item()])
        ax.axis('off')  # 关闭坐标轴
    for i in range(num_images, rows * cols):  # 将多余的子图隐藏
        fig.delaxes(axes.flatten()[i])  # 删除子图
    plt.tight_layout()  # 调整子图之间的间距
    plt.show()


class CustomMNISTDataset(Dataset):  # 自定义数据集
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        for filename in os.listdir(root):
            if filename.endswith('.png'):
                label = int(filename.split('_')[0])  # 从文件名中提取标签
                self.images.append(os.path.join(root, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Data transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load custom datasets
train_custom_data = CustomMNISTDataset(root=train_custom_path, transform=transform)
value_custom_data = CustomMNISTDataset(root=value_custom_path, transform=transform)

# Data loaders for custom datasets
BATCH_SIZE = 256  # 批次大小
train_custom_loader = DataLoader(train_custom_data, batch_size=BATCH_SIZE, shuffle=True)
value_custom_loader = DataLoader(value_custom_data, batch_size=BATCH_SIZE)

# Load MNIST datasets
train_data = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
value_data = torchvision.datasets.MNIST(root=path, train=False, transform=transform)

# Combine custom datasets with MNIST datasets
train_data_loader = DataLoader(ConcatDataset([train_data, train_custom_data]), batch_size=BATCH_SIZE,
                               shuffle=True)  # shuffle=True表示在训练过程中数据会随机打乱顺序
value_data_loader = DataLoader(ConcatDataset([value_data, value_custom_data]), batch_size=BATCH_SIZE)

# 类别名称
classes = [str(i) for i in range(10)]


# # 显示训练集中的图像
# train_images, train_labels = next(iter(train_data_loader))  # 把train_data_loader换成train_custom_loader就是显示自定义数据集
# show_images(train_images, train_labels, classes, num_images=50)
#
# # 显示测试集中的图像
# value_images, value_labels = next(iter(value_data_loader))
# show_images(value_images, value_labels, classes, num_images=50)


# Improved Net class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 这是第一个卷积层，它将输入图像的一个通道（灰度图）转换为64个特征图。卷积核大小为3x3，且使用了1像素的填充来保持输出尺寸与输入相同。
        self.bn1 = torch.nn.BatchNorm2d(64)  # 批量归一化层，在卷积层之后，用于加速训练过程并提高模型的泛化能力。它通过标准化每个特征图的激活值，使得它们具有零均值和单位方差。
        self.lrelu1 = torch.nn.LeakyReLU()  # Leaky ReLU激活函数，是ReLU激活函数的一种变体，即使输入为负值时也能给出一个小的斜率（非零输出），有助于解决ReLU在负值区域梯度消失的问题。
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)  # 最大池化层，用于降低空间维度，减少计算量同时保持最重要的信息。它取每个2x2区域内的最大值作为输出。

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 这是第二个卷积层，它将64个特征图转换为128个特征图。卷积核大小为3x3，且使用了1像素的填充来保持输出尺寸与输入相同。
        self.bn2 = torch.nn.BatchNorm2d(128)  # 批量归一化层，在卷积层之后，用于加速训练过程并提高模型的泛化能力。它通过标准化每个特征图的激活值，使得它们具有零均值和单位方差。
        self.lrelu2 = torch.nn.LeakyReLU()  # Leaky ReLU激活函数，是ReLU激活函数的一种变体，即使输入为负值时也能给出一个小的斜率（非零输出），有助于解决ReLU在负值区域梯度消失的问题。
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)  # 最大池化层，与前一个池化层类似，但步长为1，即不改变输出尺寸。

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 这是第三个卷积层，它将128个特征图转换为256个特征图。卷积核大小为3x3，且使用了1像素的填充来保持输出尺寸与输入相同。
        self.bn3 = torch.nn.BatchNorm2d(256)  # 批量归一化层，在卷积层之后，用于加速训练过程并提高模型的泛化能力。它通过标准化每个特征图的激活值，使得它们具有零均值和单位方差。
        self.lrelu3 = torch.nn.LeakyReLU()  # Leaky ReLU激活函数，是ReLU激活函数的一种变体，即使输入为负值时也能给出一个小的斜率（非零输出），有助于解决ReLU在负值区域梯度消失的问题。
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)  # 最大池化层，与前一个池化层类似，但步长为1，即不改变输出尺寸。

        self.flatten = torch.nn.Flatten()  # 展平层，将卷积层输出转换为一维向量，以便后续输入到全连接层处理。

        self.fc1 = torch.nn.Linear(256 * 6 * 6, 1024)  # 全连接层，将展平后的特征图转换为1024维向量。
        self.dropout1 = torch.nn.Dropout(0.5)  # Dropout层，用于防止过拟合，随机丢弃部分神经元，避免网络过强依赖某些神经元。
        self.lrelu4 = torch.nn.LeakyReLU()  # Leaky ReLU激活函数，是ReLU激活函数的一种变体，即使输入为负值时也能给出一个小的斜率（非零输出），有助于解决ReLU在负值区域梯度消失的问题。

        self.fc2 = torch.nn.Linear(1024, 256)  # 全连接层，将展平后的特征图转换为256维向量。
        self.dropout2 = torch.nn.Dropout(0.5)  # Dropout层，用于防止过拟合，随机丢弃部分神经元，避免网络过强依赖某些神经元。
        self.lrelu5 = torch.nn.LeakyReLU()  # Leaky ReLU激活函数，是ReLU激活函数的一种变体，即使输入为负值时也能给出一个小的斜率（非零输出），有助于解决ReLU在负值区域梯度消失的问题。

        self.fc3 = torch.nn.Linear(256, 10)  # 全连接层，将展平后的特征图转换为10维向量，用于分类。

    def forward(self, x):
        # 通过网络的前向传播过程
        # 卷积和池化: 每个卷积层后面紧跟着批量归一化和Leaky ReLU激活，然后是最大池化。
        # 这一步步减小输入的空间尺寸，同时增加特征的数量。
        # torch.Size([256, 32, 14, 14])
        x = self.pool1(self.lrelu1(self.bn1(self.conv1(x))))
        # torch.Size([256, 64, 5, 5])
        x = self.pool2(self.lrelu2(self.bn2(self.conv2(x))))
        # torch.Size([256, 128, 2, 2])
        x = self.pool3(self.lrelu3(self.bn3(self.conv3(x))))

        # 展平特征图
        # 在所有卷积和池化层之后，使用self.flatten将数据从二维特征图转换成一维向量。
        x = self.flatten(x)  # 展平后的特征图大小为256*6*6

        # 全连接层
        # 通过一系列全连接层（如fc1, fc2, fc3），逐步减少特征维度，最后得到分类结果。
        # 将展平后的特征图转换为10维向量，其中每个元素表示一个类别的概率。
        # 每层全连接后都伴随有Dropout和Leaky ReLU激活，以增强模型的非线性表达能力和防止过拟合。
        # 经过第一层全连接、dropout和激活函数
        x = self.dropout1(self.lrelu4(self.fc1(x)))
        # 经过第二层全连接、dropout和激活函数
        x = self.dropout2(self.lrelu5(self.fc2(x)))

        # 输出层
        # 最终的self.fc3是一个从256维特征映射到10维输出的全连接层
        x = self.fc3(x)
        return x


# Define the network, loss function, optimizer, and scheduler
net = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()  # 损失函数，交叉熵损失函数
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)  # 优化器，AdamW优化器 优化网络参数，使得网络在训练过程中能够更好地拟合训练数据
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率调度器，每5个epoch，学习率乘以0.5

# Training and evaluation loop
history = {'value Loss': [], 'value Accuracy': []}
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    net.train()  # 设置网络为训练模式
    process_bar = tqdm(train_data_loader, unit='step')  # 初始化进度条
    for step, (images, labels) in enumerate(process_bar):
        images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备
        optimizer.zero_grad()  # 清零梯度
        outputs = net(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        predictions = torch.argmax(outputs, dim=1)  # 预测结果
        accuracy = torch.sum(predictions == labels).item() / labels.size(0)  # 计算准确率
        process_bar.set_description(f"[{epoch}/{EPOCHS}] Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")  # 更新进度条描述

    # # 显示部分训练图像 (可选)
    # if epoch % 5 == 0:  # 每5个epoch显示一次图像
    #     train_images, train_labels = next(iter(train_data_loader))
    #     show_images(train_images, train_labels, classes)

    scheduler.step()  # 更新学习率

    # Evaluation
    net.eval()  # 设置网络为评估模式
    correct, total_loss = 0, 0
    with torch.no_grad():  # 关闭梯度计算
        for images, labels in value_data_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备
            outputs = net(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()  # 累加损失
            predictions = torch.argmax(outputs, dim=1)  # 预测结果
            correct += torch.sum(predictions == labels).item()  # 累加正确预测数

    # 计算评估集的准确率和损失
    value_accuracy = correct / len(ConcatDataset([value_data, value_custom_data]))  # 计算准确率
    value_loss = total_loss / len(value_data_loader)
    # 记录历史数据
    history['value Loss'].append(value_loss)
    history['value Accuracy'].append(value_accuracy)
    # 输出当前epoch的评估结果
    print(f"value Loss: {value_loss:.4f}, value Accuracy: {value_accuracy:.4f}")

# 绘制损失和准确率曲线
# Plot the value loss and accuracy
plt.plot(history['value Loss'], label='value Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history['value Accuracy'], color='red', label='value Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Save the model
torch.save(net.state_dict(), './model_optimized_with_custom_data_5.pth')
