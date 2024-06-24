import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'GTK3Agg' 等
# import matplotlib.pyplot as plt

from input_drawings.input_2 import App

if __name__ == "__main__":
    root = tk.Tk()  # 创建一个窗口
    app = App(root)  # 创建一个App对象
    root.mainloop()  # 进入主循环 监听事件


