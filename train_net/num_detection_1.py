import torch
import torchvision
from tqdm import tqdm

# from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'GTK3Agg' 等
import matplotlib.pyplot as plt

# 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

path = '../data/'  # 数据集下载后保存的目录

# 下载训练集和测试集
trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST(path, train=False, transform=transform)

# 设定每一个Batch的大小
BATCH_SIZE = 256

# 构建数据集和测试集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)


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


net = Net()
# 将模型转换到device中，并将其结构显示出来
# print(net.to(device))

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

EPOCHS = 10
# 存储训练过程
history = {'Test Loss': [], 'Test Accuracy': []}

for epoch in range(1, EPOCHS + 1):
    # 训练部分
    processBar = tqdm(trainDataLoader, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

        # 测试部分
        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            net.train(False)
            for testImgs, labels in testDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)

                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
            testLoss = totalLoss / len(testDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()

# 对测试Loss进行可视化
plt.plot(history['Test Loss'], label='Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 对测试准确率进行可视化
plt.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 　保存模型
torch.save(net, '../model/model_1.pth')
