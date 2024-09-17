import numpy as np
import pandas as pd
import cv2
import sys
import matplotlib.pyplot as plt
import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

constant = 15

label = None
brightness_label = None
original_matrix = None

# 題目選項
option = 1

# 讀取檔案的函數
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 讀取每個 .64 檔案
file_path_LISA = "/Users/666jackson/Desktop/實驗室/image_processing/image/LISA.64"
file_path_JET = "/Users/666jackson/Desktop/實驗室/image_processing/image/JET.64"
file_path_LIBERTY = "/Users/666jackson/Desktop/實驗室/image_processing/image/LIBERTY.64"
file_path_LINCOLN = "/Users/666jackson/Desktop/實驗室/image_processing/image/LINCOLN.64"

string_LISA = read_file(file_path_LISA)
string_JET = read_file(file_path_JET)
string_LIBERTY = read_file(file_path_LIBERTY)
string_LINCOLN = read_file(file_path_LINCOLN)

string_data = string_LINCOLN
string_data2 = string_LISA

# 定義轉換函數
def string_to_matrix(string_data):
    matrix = np.zeros((64, 64), dtype=int)
    i = 0
    if i < len(string_data):
        for x in range(64):
            for y in range(65):
                word = string_data[i]
                if 'A' <= word <= 'Z':
                    num = ord(word) - ord('A') + 10
                    matrix[x][y] = num
                elif word.isdigit():
                    num = int(word)
                    matrix[x][y] = num
                i += 1

    return matrix

# 將每個字串轉換為矩陣
matrix_LISA = string_to_matrix(string_LISA)
matrix_JET = string_to_matrix(string_JET)
matrix_LIBERTY = string_to_matrix(string_LIBERTY)
matrix_LINCOLN = string_to_matrix(string_LINCOLN)


# 輸出以確認轉換結果（顯示前 5 行）
'''
print("LISA.64 Matrix (前5行):\n", matrix_LISA)
print("JET.64 Matrix (前5行):\n", matrix_JET[:5])
print("LIBERTY.64 Matrix (前5行):\n", matrix_LIBERTY[:5])
print("LINCOLN.64 Matrix (前5行):\n", matrix_LINCOLN[:5])'''

# 顯示矩陣作為影像並保存到文件
def display_matrix_as_image(matrix, file_name):
    matrix_8bit = np.uint8((matrix / 31.0) * 255)  # 將 0-31 的數值範圍縮放到 0-255

    # 將影像保存到文件
    cv2.imwrite(file_name, matrix_8bit)
    print(f"Image saved as {file_name}")


'''# 保存影像到文件
display_matrix_as_image(matrix_LISA, "output_image_LISA1.png")
display_matrix_as_image(matrix_JET, "output_image_JET.png")
display_matrix_as_image(matrix_LIBERTY, "output_image_LIBERTY.png")
display_matrix_as_image(matrix_LINCOLN, "output_image_LINCOLN.png")
'''

def pandas_to_csv(matrix):
    # 創建一個 32 個元素的直方圖計數數組
    histogram = np.zeros(32, dtype=int)

    # 計算每個灰階值出現的次數
    for row in matrix:
        for value in row:
            if 0 <= value <= 31:
                histogram[value] += 1

    # 創建 DataFrame
    df = pd.DataFrame({
        '灰階值': range(32),
        '數量': histogram
    })
    return df

# 統計像素值表格
pandas_LISA = pandas_to_csv(matrix_LISA)
pandas_JET = pandas_to_csv(matrix_JET)
pandas_LIBERTY = pandas_to_csv(matrix_LIBERTY)
pandas_LINCOLN = pandas_to_csv(matrix_LINCOLN)

#print(pandas_LINCOLN)
#print(pandas_LISA)

'''
def write_histogram_to_csv(filename, matrix):
    # 創建一個 32 個元素的直方圖計數數組
    histogram = np.zeros(32, dtype=int)

    # 計算每個灰階值出現的次數
    for row in matrix:
        for value in row:
            if 0 <= value <= 31:
                histogram[value] += 1

    # 創建 DataFrame
    df = pd.DataFrame({
        '灰階值': range(32),
        '數量': histogram
    })

    # 將 DataFrame 寫入 CSV 文件
    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"直方圖數據已成功寫入 CSV 文件: {filename}")
    except IOError:
        print(f"無法打開文件: {filename}")

# 寫入csv檔並輸出

write_histogram_to_csv('histogram_LISA.csv', matrix_LISA)
write_histogram_to_csv('histogram_JET.csv', matrix_JET)
write_histogram_to_csv('histogram.LIBERTY.csv', matrix_LIBERTY)
write_histogram_to_csv('histogramLINCOLN.csv', matrix_LINCOLN)
'''
def plot_histogram(data):

    app = QApplication(sys.argv)
    window = QMainWindow()

    # 創建中心部件和布局
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    window.setCentralWidget(central_widget)

    # Matplotlib 圖表
    figure, ax = plt.subplots()
    canvas = FigureCanvas(figure)

    # 繪製直方圖
    ax.bar(data['灰階值'], data['數量'], color='skyblue')
    ax.set_xlabel('Gray Level')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram')

    # 將圖表添加到布局中
    layout.addWidget(canvas)
    # 設置窗口標題
    window.setWindowTitle("Histogram Viewer")
    # 設置窗口大小
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

# 繪製直方圖和顯示圖像
def plot_histogram_and_image(matrix, data, filename):
    app = QApplication(sys.argv)
    window = QMainWindow()

    # 創建中心部件和佈局
    central_widget = QWidget()
    layout = QHBoxLayout(central_widget)
    window.setCentralWidget(central_widget)

    # 繪製直方圖
    figure, ax = plt.subplots()
    canvas = FigureCanvas(figure)
    ax.bar(data['灰階值'], data['數量'], color='skyblue')
    ax.set_xlabel('Gray Level')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram')

    # 添加直方圖到佈局
    layout.addWidget(canvas)

    # 顯示圖像
    image_data = display_matrix_as_image(matrix,filename)
    # image_path = "temp_image.png"
    # cv2.imwrite(image_path, image_data)  # 將圖像保存為文件

    label = QLabel()
    pixmap = QPixmap(filename)

    # 將影像放大兩倍，並保持比例
    pixmap_scaled = pixmap.scaled(pixmap.width() * 6, pixmap.height() * 6, Qt.KeepAspectRatio)
    label.setPixmap(pixmap_scaled)
    label.setAlignment(Qt.AlignCenter)

    # 添加圖像到佈局
    layout.addWidget(label)
    window.setWindowTitle("Image and Histogram Viewer")
    window.resize(1200, 600)
    window.show()
    app.exec_()
    #sys.exit(app.exec_())

# 加法
def Add_image(string_data, constant):
    matrix = np.zeros((64, 64), dtype=int)
    i = 0
    if i < len(string_data):
        for x in range(64):
            for y in range(65):
                word = string_data[i]
                if 'A' <= word <= 'Z':
                    num = ord(word) - ord('A') + 10 + constant
                    matrix[x][y] = num
                elif word.isdigit():
                    num = int(word) + constant
                    matrix[x][y] = num
                i += 1

    matrix = np.clip(matrix, 0, 31)
    return matrix

# 減法
def Sub_image(string_data, constant):
    matrix = np.zeros((64, 64), dtype=int)
    i = 0
    if i < len(string_data):
        for x in range(64):
            for y in range(65):
                word = string_data[i]
                if 'A' <= word <= 'Z':
                    num = ord(word) - ord('A') + 10 - constant
                    matrix[x][y] = num
                elif word.isdigit():
                    num = int(word) - constant
                    matrix[x][y] = num
                i += 1

    matrix = np.clip(matrix, 0, 31)
    return matrix

# 乘法
def Mul_image(string_data, constant):
    matrix = np.zeros((64, 64), dtype=int)
    i = 0
    if i < len(string_data):
        for x in range(64):
            for y in range(65):
                word = string_data[i]
                if 'A' <= word <= 'Z':
                    num = (ord(word) - ord('A') + 10) * constant
                    matrix[x][y] = num
                elif word.isdigit():
                    num = int(word) * constant
                    matrix[x][y] = num
                i += 1

    matrix = np.clip(matrix, 0, 31)
    return matrix

# 平均
def Avg_3(string_temp, string_temp2):
    matrix_ex = np.zeros((65, 65), dtype=int)
    matrix_ex2 = np.zeros((65, 65), dtype=int)
    avg = np.zeros((65, 65), dtype=float)

    i = 0
    if i < len(string_temp):
        for x in range(64):
            for y in range(65):
                # 處理第一個字串
                word1 = string_temp[i]
                if 'A' <= word1 <= 'Z':
                    num = ord(word1) - ord('A') + 10
                elif word1.isdigit():
                    num = int(word1)
                else:
                    num = 0
                matrix_ex[x][y] = num

                # 處理第二個字串
                word2 = string_temp2[i]
                if 'A' <= word2 <= 'Z':
                    num2 = ord(word2) - ord('A') + 10
                elif word2.isdigit():
                    num2 = int(word2)
                else:
                    num2 = 0
                matrix_ex2[x][y] = num2

                # 計算平均值
                avg[x][y] = (matrix_ex[x][y] + matrix_ex2[x][y]) / 2.0
                i += 1

    avg = np.clip(avg, 0, 31)
    return avg

def function(string_temp):
    f = np.zeros((65, 65), dtype=int)
    g = np.zeros((65, 65), dtype=int)

    i = 0
    if i < len(string_temp):
        for x in range(64):
            for y in range(65):  # 修正範圍為 64x64
                word1 = string_temp[i]

                # 處理字母
                if 'A' <= word1 <= 'Z':
                    num = ord(word1) - ord('A') + 10
                # 處理數字
                elif word1.isdigit():
                    num = int(word1)
                else:
                    num = 0

                f[x][y] = num

                # 計算 g(x, y) = f(x, y) - f(x-1, y)
                if x > 0:  # 需要檢查 x 是否大於 0，避免索引超出範圍
                    g[x][y] = f[x][y] - f[x-1][y]
                else:
                    g[x][y] = f[x][y]  # 若 x = 0，則 g[x][y] 設為 f[x][y]

                i += 1

    g = np.clip(g, 0, 31)
    return g

# 讀取並更新影像顯示
def update_image(matrix):
    matrix_8bit = np.uint8((matrix / 31.0) * 255)  # 將 0-31 範圍縮放到 0-255
    height, width = matrix_8bit.shape
    q_img = QImage(matrix_8bit.data, width, height, width, QImage.Format_Grayscale8)
    pixmap = QPixmap(q_img).scaled(300, 300)  # 調整顯示大小
    label.setPixmap(pixmap)

# 調整亮度的回調函數
def change_brightness(value):
    if option == 1 :
        adjusted_matrix = Add_image(string_data, value)
    elif option == 2 :
        adjusted_matrix = Sub_image(string_data, value)
    elif option == 3 :
        adjusted_matrix = Mul_image(string_data, value)
    elif option == 4 :
        adjusted_matrix = Avg_3(string_data, string_data2)
    elif option == 5 :
        adjusted_matrix = function(string_data)

    update_image(adjusted_matrix)

# 創建界面
def create_window(option):
    global label, original_matrix, brightness_label

    # 初始化原始影像矩陣
    if option == 1 :
        original_matrix = Add_image(string_data, 0)
    elif option == 2 :
        original_matrix = Sub_image(string_data, 0)
    elif option == 3 :
        original_matrix = Mul_image(string_data, 0)
    elif option == 4 :
        original_matrix = Avg_3(string_data,string_data2)
    elif option == 5 :
        original_matrix = function(string_data)

    # 創建主窗口
    window = QMainWindow()

    # 創建 QLabel 用於顯示影像
    label = QLabel()
    update_image(original_matrix)

    # 創建 QLabel 用於顯示亮度數值
    brightness_label = QLabel("Brightness Adjustment: 0")

    # 創建 QSlider 用於控制亮度
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)  # 設置亮度調整的範圍
    slider.setMaximum(32)
    slider.setValue(0)  # 初始值 0 表示不改變亮度
    slider.setTickPosition(QSlider.TicksBelow)  # 在滑桿下方顯示刻度
    slider.setTickInterval(10)  # 每 10 單位顯示一個刻度
    slider.valueChanged.connect(change_brightness)

    # 創建垂直佈局
    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addWidget(slider)

    # 創建中心小部件並設置佈局
    central_widget = QWidget()
    central_widget.setLayout(layout)
    window.setCentralWidget(central_widget)

    # 設置窗口標題和大小
    window.setWindowTitle('Brightness Adjustment with Slider')
    window.resize(600, 700)

    return window

# 主程序
def main(option):
    app = QApplication(sys.argv)
    window = create_window(option)
    window.show()
    app.exec_()
    # sys.exit(app.exec_())


if __name__ == "__main__":
    while option <= 5 :
        main(option)
        option += 1

    plot_histogram_and_image(matrix_LISA, pandas_LISA, 'LISA.png')
    plot_histogram_and_image(matrix_JET, pandas_JET, 'JET.png')
    plot_histogram_and_image(matrix_LINCOLN, pandas_LINCOLN, 'LINCOLN.png')
    plot_histogram_and_image(matrix_LIBERTY, pandas_LIBERTY, 'LIBERTY.png')


