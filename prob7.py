import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLineEdit, QPushButton, QScrollArea, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt5.QtCore import Qt

# 將 numpy 陣列轉換為 QPixmap 以顯示
def numpy_to_pixmap(image, is_color=False):
    if is_color:
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    return QPixmap.fromImage(q_image)

def manual_absdiff(gray1, gray2):
    # 確保兩張圖像的大小相同
    if gray1.shape != gray2.shape:
        raise ValueError("Both images must have the same dimensions.")

    # 手動計算絕對差值
    height, width = gray1.shape
    diff_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # 計算絕對差值，並確保在 [0, 255] 範圍內
            diff_image[y, x] = abs(int(gray1[y, x]) - int(gray2[y, x]))

    return diff_image

# 手動計算灰階圖片的直方圖
def calculate_histogram(image):
    histogram = [0] * 256  # 初始化長度為 256 的直方圖，代表每個灰階值的計數

    # 遍歷每個像素並計算灰階值的頻率
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            gray_value = image[y, x]
            histogram[gray_value] += 1

    return histogram

# 手動閾值轉換函數，將灰階圖片轉換為二值圖片
def manual_threshold(image, threshold):
    # 創建一個與原圖同樣大小的空白二值圖
    binary_image = np.zeros_like(image, dtype=np.uint8)

    # 根據閾值進行二值化轉換
    binary_image[image >= threshold] = 255
    binary_image[image < threshold] = 0

    return binary_image

# 將直方圖轉換為圖片
def histogram_to_pixmap(histogram, width=300, height=200):
    # 創建一個空白的QImage作為繪製直方圖的畫布
    histogram_image = QImage(width, height, QImage.Format_RGB32)
    histogram_image.fill(QColor('white'))  # 背景填充為白色

    # 使用QPainter在QImage上繪製直方圖
    painter = QPainter(histogram_image)
    max_value = max(histogram)  # 獲取最大頻率值，用於規模化顯示
    bar_width = width // len(histogram)  # 每個條的寬度

    for i, count in enumerate(histogram):
        if count > 0:
            # 計算條的高度
            bar_height = int((count / max_value) * (height - 10))
            # 設置顏色
            painter.setBrush(QColor('black'))
            # 繪製條形圖
            painter.drawRect(i * bar_width, height - bar_height, bar_width - 1, bar_height)

    painter.end()
    return QPixmap.fromImage(histogram_image)

# 更新二值圖片並顯示
def update_binary_image(label, image, threshold, display_size):
    binary_image = manual_threshold(image, threshold)
    binary_image_resized = cv2.resize(binary_image, display_size)
    label.setPixmap(numpy_to_pixmap(binary_image_resized))

# 設置閾值變更時的更新行為
def on_threshold_change(value, label_binary, gray_image, display_size):
    update_binary_image(label_binary, gray_image, value, display_size)

# 設置手動輸入閾值時的更新行為
def on_threshold_input(input_field, slider, label_binary, gray_image, display_size):
    try:
        threshold = int(input_field.text())
        if threshold < 0 or threshold > 255:
            raise ValueError("Threshold must be between 0 and 255.")
    except ValueError:
        threshold = 128
        input_field.setText(str(threshold))

    slider.setValue(threshold)
    update_binary_image(label_binary, gray_image, threshold, display_size)

# 手動雙線性內插法來調整解析度（放大或縮小圖片）
def adjust_resolution(image, scale):
    original_height, original_width = image.shape[:2]
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    # 建立新圖片矩陣
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # 原始圖片中的對應點座標
            original_x = x / scale
            original_y = y / scale

            # 找到四個相鄰像素的座標（上左、上右、下左、下右）
            x1 = int(original_x)
            x2 = min(x1 + 1, original_width - 1)
            y1 = int(original_y)
            y2 = min(y1 + 1, original_height - 1)

            # 四個點的灰階值
            Q11 = image[y1, x1]
            Q21 = image[y1, x2]
            Q12 = image[y2, x1]
            Q22 = image[y2, x2]

            # 計算水平和垂直的權重
            r1 = (x2 - original_x) * Q11 + (original_x - x1) * Q21
            r2 = (x2 - original_x) * Q12 + (original_x - x1) * Q22
            P = (y2 - original_y) * r1 + (original_y - y1) * r2

            # 設置新像素值
            resized_image[y, x] = int(P)

    return resized_image

# 調整灰階級別
def adjust_grayscale_levels(image, levels):
    # 計算每個新灰階級別的區間範圍
    if levels < 1:
        levels = 1
    interval = 256 / levels
    adjusted_image = np.floor(image / interval) * interval
    return adjusted_image.astype(np.uint8)

# 亮度和對比度調整函數：通過調整每個像素的值來實現亮度和對比度的變化。
# 亮度調整公式為 output = input + brightness，對比度調整公式為 output = (input - 128) * contrast + 128。

# 調整亮度的函數
def adjust_brightness(image, brightness):
    #print(f"Original image min: {image.min()}, max: {image.max()}, dtype: {image.dtype}")

    # 將亮度值增加到像素值，並使用 np.clip 限制範圍在 [0, 255] 之間
    adjusted_image = image.astype(np.int16) + brightness  # 使用 int16 以避免 uint8 溢出
    adjusted_image = np.clip(adjusted_image, 0, 255)  # 將所有值限制在 [0, 255]

    # 檢查調整後的圖像數據範圍和類型
    # print(f"Adjusted image min: {adjusted_image.min()}, max: {adjusted_image.max()}, dtype: {adjusted_image.dtype}")

    return adjusted_image.astype(np.uint8)

# 調整對比度
def adjust_contrast(image, contrast):
    # 根據對比度調整每個像素的值，並裁剪到合法範圍
    adjusted_image = np.clip(128 + (image - 128) * contrast, 0, 255)
    return adjusted_image.astype(np.uint8)

# 更新解析度
def apply_resolution(label, image, scale_percentage, display_size):
    scale = scale_percentage / 100.0
    adjusted_image = adjust_resolution(image, scale)
    adjusted_image_resized = cv2.resize(adjusted_image, display_size)
    label.setPixmap(numpy_to_pixmap(adjusted_image_resized))

# 更新灰階級別
def apply_grayscale(label, image, levels, display_size):
    adjusted_image = adjust_grayscale_levels(image, levels)
    adjusted_image_resized = cv2.resize(adjusted_image, display_size)
    label.setPixmap(numpy_to_pixmap(adjusted_image_resized))

# 更新亮度
def apply_brightness(label, image, brightness, display_size):
    adjusted_image = adjust_brightness(image, brightness)
    adjusted_image_resized = cv2.resize(adjusted_image, display_size)
    label.setPixmap(numpy_to_pixmap(adjusted_image_resized))

# 更新對比度
def apply_contrast(label, image, contrast, display_size):
    adjusted_image = adjust_contrast(image, contrast)
    adjusted_image_resized = cv2.resize(adjusted_image, display_size)
    label.setPixmap(numpy_to_pixmap(adjusted_image_resized))


# 直方圖均衡化
def histogram_equalization(image):

    height, width = image.shape

    # 計算直方圖
    hist = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        hist[pixel] += 1

    # 計算累積分佈函數 (CDF)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # 正規化 CDF
    cdf_min = cdf[cdf > 0].min()  # 找到非零最小值
    cdf_max = cdf[-1]  # 最大的 CDF 值

    # 將 CDF 映射到 [0, 255]
    cdf_mapped = np.floor((cdf - cdf_min) * 255 / (cdf_max - cdf_min)).astype(np.uint8)

    # 使用映射的 CDF 重映射圖像
    equalized_image = cdf_mapped[image]

    return equalized_image



# 更新直方圖均衡化圖片
def apply_histogram_equalization(label, image, display_size):
    adjusted_image = histogram_equalization(image)
    adjusted_image_resized = cv2.resize(adjusted_image, display_size)
    label.setPixmap(numpy_to_pixmap(adjusted_image_resized))


def main(image_color, gray1, gray2, image_diff, display_size=(400, 400)):
    app = QApplication(sys.argv)

    window = QWidget()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)  # 啟用視窗大小調整

    content_widget = QWidget()
    main_layout = QVBoxLayout(content_widget)

    # 顯示彩色圖片和兩張灰階圖片
    image_layout = QVBoxLayout()
    image_color_resized = cv2.resize(image_color, display_size)
    label_color = QLabel()
    label_color.setPixmap(numpy_to_pixmap(image_color_resized, is_color=True))

    image_layout_gray = QHBoxLayout()
    gray1_resized = cv2.resize(gray1, display_size)
    gray2_resized = cv2.resize(gray2, display_size)
    image_diff_resized = cv2.resize(image_diff, display_size)

    label_gray1 = QLabel()
    label_gray1.setPixmap(numpy_to_pixmap(gray1_resized))
    label_gray2 = QLabel()
    label_gray2.setPixmap(numpy_to_pixmap(gray2_resized))
    label_diff = QLabel()
    label_diff.setPixmap(numpy_to_pixmap(image_diff_resized))

    image_layout.addWidget(label_color)
    image_layout_gray.addWidget(label_gray1)
    image_layout_gray.addWidget(label_gray2)
    image_layout_gray.addWidget(label_diff)

    # 顯示每張灰階圖片的直方圖
    histogram_layout = QHBoxLayout()
    histogram1 = calculate_histogram(gray1)
    histogram2 = calculate_histogram(gray2)
    histogram_diff = calculate_histogram(image_diff)

    label_hist1 = QLabel("Histogram of Gray Image 1")
    label_hist1.setPixmap(histogram_to_pixmap(histogram1))
    label_hist2 = QLabel("Histogram of Gray Image 2")
    label_hist2.setPixmap(histogram_to_pixmap(histogram2))
    label_hist_diff = QLabel("Histogram of Difference Image")
    label_hist_diff.setPixmap(histogram_to_pixmap(histogram_diff))

    histogram_layout.addWidget(label_hist1)
    histogram_layout.addWidget(label_hist2)
    histogram_layout.addWidget(label_hist_diff)

    # 手動閾值轉換的控件
    threshold_layout = QHBoxLayout()

    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(255)
    slider.setValue(128)

    input_field = QLineEdit(str(slider.value()))
    input_field.setFixedWidth(50)

    btn_apply = QPushButton("Apply Threshold")
    btn_apply.clicked.connect(lambda: on_threshold_input(input_field, slider, label_binary, gray1, display_size))

    # 設置 slider 和輸入框的事件連接
    slider.valueChanged.connect(lambda value: on_threshold_change(value, label_binary, gray1, display_size))
    input_field.returnPressed.connect(lambda: on_threshold_input(input_field, slider, label_binary, gray1, display_size))

    threshold_layout.addWidget(QLabel("Threshold:"))
    threshold_layout.addWidget(slider)
    threshold_layout.addWidget(input_field)
    threshold_layout.addWidget(btn_apply)

    # 顯示二值化後的圖片
    label_binary = QLabel("Binary Image")
    update_binary_image(label_binary, gray1, slider.value(), display_size)

    # 添加解析度和灰階級別調整控件
    resolution_layout = QHBoxLayout()
    resolution_slider = QSlider(Qt.Horizontal)
    resolution_slider.setMinimum(10)
    resolution_slider.setMaximum(200)
    resolution_slider.setValue(100)  # 預設為 100% 大小

    resolution_label = QLabel("Resolution: 100%")
    resolution_slider.valueChanged.connect(lambda value: resolution_label.setText(f"Resolution: {value}%"))

    btn_apply_resolution = QPushButton("Apply Resolution")
    btn_apply_resolution.clicked.connect(lambda: apply_resolution(label_adjusted, gray1, resolution_slider.value(), display_size))

    resolution_layout.addWidget(resolution_label)
    resolution_layout.addWidget(resolution_slider)
    resolution_layout.addWidget(btn_apply_resolution)

    # 灰階級別調整
    grayscale_layout = QHBoxLayout()
    grayscale_slider = QSlider(Qt.Horizontal)
    grayscale_slider.setMinimum(1)
    grayscale_slider.setMaximum(256)
    grayscale_slider.setValue(256)  # 預設為 256 個灰階級別

    grayscale_label = QLabel("Grayscale Levels: 256")
    grayscale_slider.valueChanged.connect(lambda value: grayscale_label.setText(f"Grayscale Levels: {value}"))

    btn_apply_grayscale = QPushButton("Apply Grayscale Levels")
    btn_apply_grayscale.clicked.connect(lambda: apply_grayscale(label_adjusted, gray1, grayscale_slider.value(), display_size))

    grayscale_layout.addWidget(grayscale_label)
    grayscale_layout.addWidget(grayscale_slider)
    grayscale_layout.addWidget(btn_apply_grayscale)

    # 亮度調整
    brightness_layout = QHBoxLayout()
    brightness_slider = QSlider(Qt.Horizontal)
    brightness_slider.setMinimum(-100)
    brightness_slider.setMaximum(100)
    brightness_slider.setValue(0)  # 初始亮度為 0

    brightness_label = QLabel("Brightness: 0")
    brightness_slider.valueChanged.connect(lambda value: brightness_label.setText(f"Brightness: {value}"))

    btn_apply_brightness = QPushButton("Apply Brightness")
    btn_apply_brightness.clicked.connect(lambda: apply_brightness(label_adjusted, gray1, brightness_slider.value(), display_size))

    brightness_layout.addWidget(brightness_label)
    brightness_layout.addWidget(brightness_slider)
    brightness_layout.addWidget(btn_apply_brightness)

    # 對比度調整
    contrast_layout = QHBoxLayout()
    contrast_slider = QSlider(Qt.Horizontal)
    contrast_slider.setMinimum(1)
    contrast_slider.setMaximum(10)
    contrast_slider.setValue(1)  # 初始對比度為 1

    contrast_label = QLabel("Contrast: 1.0")
    contrast_slider.valueChanged.connect(lambda value: contrast_label.setText(f"Contrast: {value / 1.0:.1f}"))

    btn_apply_contrast = QPushButton("Apply Contrast")
    btn_apply_contrast.clicked.connect(lambda: apply_contrast(label_adjusted, gray1, contrast_slider.value(), display_size))

    contrast_layout.addWidget(contrast_label)
    contrast_layout.addWidget(contrast_slider)
    contrast_layout.addWidget(btn_apply_contrast)

    # 直方圖均衡化
    btn_histogram_equalization = QPushButton("Apply Histogram Equalization")
    btn_histogram_equalization.clicked.connect(lambda: apply_histogram_equalization(label_adjusted, gray1, display_size))

    # 顯示調整後的圖片
    label_adjusted = QLabel("Adjusted Image")
    label_adjusted.setPixmap(numpy_to_pixmap(gray1))  # 初始顯示灰階圖片1

    # 組合
    main_layout.addLayout(image_layout)  # 原圖
    main_layout.addLayout(image_layout_gray)  # 灰階圖
    main_layout.addLayout(histogram_layout)  # 直方圖
    main_layout.addLayout(threshold_layout)  # 閾值控件
    main_layout.addWidget(label_binary)  # 二值化圖片
    main_layout.addLayout(resolution_layout)  # 解析度調整
    main_layout.addLayout(grayscale_layout)  # 灰階級別調整
    main_layout.addLayout(brightness_layout)  # 亮度調整
    main_layout.addLayout(contrast_layout)  # 對比度調整
    main_layout.addWidget(btn_histogram_equalization)  # 直方圖均衡化
    main_layout.addWidget(label_adjusted)  # 調整後的圖片

    # 將內容設置為 QScrollArea 的窗口小部件
    scroll.setWidget(content_widget)

    # 設置主視窗佈局
    window_layout = QVBoxLayout(window)
    window_layout.addWidget(scroll)
    window.resize(2000, 1000)
    window.setWindowTitle('Image and Histogram Viewer with Thresholding and Adjustments')
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    image_path = 'Lenna.bmp'

    # 讀取彩色圖片
    image_color = cv2.imread(image_path)
    image_color_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)  # 將BGR轉換為RGB
    image_np = np.array(image_color)

    B, G, R = cv2.split(image_color)

    gray1 = np.mean(image_np, axis=2)
    gray1 = gray1.astype(np.uint8)
    gray2 = 0.299 * image_np[:,:,0]+ 0.587 * image_np[:,:,1] + 0.114 * image_np[:,:,2]
    gray2 = gray2.astype(np.uint8)

    # 計算兩張灰階圖片的差異
    #image_diff = cv2.absdiff(gray1, gray2)
    image_diff = manual_absdiff(gray1, gray2)

    main(image_color_rgb, gray1, gray2, image_diff, display_size=(300, 300))

    print('ddd')