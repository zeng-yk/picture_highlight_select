import sys
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QSpinBox, QGroupBox, QFileDialog, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QRectF, QPoint


class ScaleRuler(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.height_ = 40
        self.setFixedHeight(self.height_)
        self.ruler_start = 10
        self.ruler_end = 150  # 初始比例尺长度（像素）
        self.max_display_length = 800  # 增加最大显示长度，允许更长的比例尺
        self.dragging = False
        self.dragging_start = False  # 标记是否正在拖动左侧端点
        self.ruler_value = 1.0  # 真实长度（用户定义）
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        # 允许整个比例尺控件在窗口中自由移动
        self.ruler_dragging = False
        self.drag_start_pos = None
        # 设置初始位置在图片区域中间位置
        self.setGeometry(300, 50, 500, self.height_)  # 更宽的初始宽度
        self.setMinimumWidth(200)  # 设置最小宽度
        self.setMaximumWidth(2000)  # 设置最大宽度，允许更长的比例尺

    def paintEvent(self, event):
        # 使用双缓冲绘制减少闪烁
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)  # 启用抗锯齿
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)  # 平滑像素图变换

        # 设置画笔
        pen = QPen(QColor(0, 206, 209), 2)  # 蓝色，更易于观察
        painter.setPen(pen)

        # 绘制比例尺主线
        y = self.height_ // 2
        painter.drawLine(self.ruler_start, y, self.ruler_end, y)

        # 绘制刻度，限制最多显示10个分度值
        ruler_length = self.ruler_end - self.ruler_start

        # 确保至少有两个刻度点，最多10个
        if ruler_length > 20:  # 确保有足够的空间显示刻度
            # 限制最多显示10个刻度
            max_ticks = 10
            num_ticks = min(max_ticks, max(2, int(ruler_length / 30) + 1))
            tick_positions = [self.ruler_start + i * (ruler_length / (num_ticks - 1)) for i in range(num_ticks)]

            # 设置字体一次，避免重复设置
            painter.setFont(QFont('Arial', 8))

            for i, x in enumerate(tick_positions):
                x_pos = int(x)  # 转换为整数，避免浮点数绘制问题
                painter.drawLine(x_pos, y - 5, x_pos, y + 5)
                # 计算每个刻度对应的实际值
                value = (i / (num_ticks - 1)) * self.ruler_value
                painter.drawText(QPoint(x_pos - 10, y - 8), f"{value:.1f}")
        else:
            # 如果比例尺太短，至少显示起点和终点
            painter.setFont(QFont('Arial', 8))
            painter.drawLine(self.ruler_start, y - 5, self.ruler_start, y + 5)
            painter.drawLine(self.ruler_end, y - 5, self.ruler_end, y + 5)
            painter.drawText(QPoint(self.ruler_start - 10, y - 8), "0.0")
            painter.drawText(QPoint(self.ruler_end - 10, y - 8), f"{self.ruler_value:.1f}")

        # 绘制总长值
        painter.drawText(QPoint(self.ruler_start, y + 20), f"{self.ruler_value:.1f} nm")

    def mousePressEvent(self, event):
        if abs(event.x() - self.ruler_end) < 10:
            # 右侧端点拖动
            self.dragging = True
            self.setCursor(Qt.ClosedHandCursor)
        elif abs(event.x() - self.ruler_start) < 10:
            # 左侧端点拖动
            self.dragging_start = True
            self.setCursor(Qt.ClosedHandCursor)
        elif self.ruler_start <= event.x() <= self.ruler_end:
            # 如果点击在比例尺线上，但不是端点，则开始整个比例尺的拖动
            if event.button() == Qt.RightButton:
                # 右键点击弹出设置真实长度对话框
                val, ok = QInputDialog.getDouble(self, "设置比例尺值", "当前比例尺对应的真实值：",
                                                 value=self.ruler_value, min=0.01)
                if ok:
                    self.ruler_value = val
                    self.update()
            else:
                # 左键点击开始拖动整个比例尺
                self.ruler_dragging = True
                self.drag_start_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging:
            # 拖动右侧端点
            # 允许数值随鼠标移动而变化，不限制显示长度
            raw_end = max(self.ruler_start + 10, event.x())

            # 直接更新结束位置，不限制最大长度
            self.ruler_end = raw_end

            # 如果比例尺长度超过控件宽度，自动调整控件宽度
            ruler_length = self.ruler_end - self.ruler_start
            if ruler_length + 40 > self.width():  # 增加边距，确保有足够空间
                new_width = ruler_length + 40  # 左右各预留20像素
                self.resize(new_width, self.height_)  # 直接调整控件大小

            self.update()
        elif self.dragging_start:
            # 拖动左侧端点，只允许向右移动，不能向左移动超过初始位置
            new_start = event.x()
            # 确保不能向左拖动超过初始位置，且与右端点保持最小距离
            if 10 <= new_start < self.ruler_end - 10:  # 10是初始位置，不能向左拖动
                self.ruler_start = new_start
                self.update()
        elif self.ruler_dragging:
            # 拖动整个比例尺
            delta = event.pos() - self.drag_start_pos
            self.move(self.pos() + delta)
            self.drag_start_pos = event.pos()
            # 防止闪烁，使用update()而不是repaint()
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.dragging_start = False
        self.ruler_dragging = False
        self.drag_start_pos = None
        self.setCursor(Qt.OpenHandCursor)


class ImageViewer(QWidget):
    mouseMoved = pyqtSignal(int, int)
    mouseClicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.image = None
        self.display_image = None
        self.setMouseTracking(True)

    def setImage(self, image):
        self.image = image
        self.display_image = image.copy()
        self.update()

    def paintEvent(self, event):
        if self.display_image is not None:
            painter = QPainter(self)

            # Convert the image to QImage
            height, width, channel = self.display_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.display_image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # Scale the image to fit the widget while maintaining aspect ratio
            scaled_img = qImg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Calculate position to center the image
            x = (self.width() - scaled_img.width()) // 2
            y = (self.height() - scaled_img.height()) // 2
            painter.drawImage(x, y, scaled_img)

    def mouseMoveEvent(self, event):
        if self.display_image is not None:
            # Calculate the actual image position and size
            img_height, img_width = self.display_image.shape[:2]
            widget_width = self.width()
            widget_height = self.height()

            # Calculate scaling factor and offsets
            scale = min(widget_width / img_width, widget_height / img_height)
            scaled_width = int(img_width * scale)
            scaled_height = int(img_height * scale)
            x_offset = (widget_width - scaled_width) // 2
            y_offset = (widget_height - scaled_height) // 2

            # Convert widget coordinates to image coordinates
            img_x = int((event.x() - x_offset) / scale)
            img_y = int((event.y() - y_offset) / scale)

            # Only emit if the point is within the image
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                self.mouseMoved.emit(img_x, img_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.display_image is not None:
            # Same coordinate conversion as in mouseMoveEvent
            img_height, img_width = self.display_image.shape[:2]
            widget_width = self.width()
            widget_height = self.height()

            scale = min(widget_width / img_width, widget_height / img_height)
            scaled_width = int(img_width * scale)
            scaled_height = int(img_height * scale)
            x_offset = (widget_width - scaled_width) // 2
            y_offset = (widget_height - scaled_height) // 2

            img_x = int((event.x() - x_offset) / scale)
            img_y = int((event.y() - y_offset) / scale)

            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                self.mouseClicked.emit(img_x, img_y)


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("亮点提取工具")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.original_image = None
        self.gray_image = None
        self.threshold_image = None
        self.processed_image = None
        self.contours = []
        self.centers = []
        self.center_to_contour = {}
        self.used_centers = set()
        self.saved_connections = []

        self.hovered_contour = None
        self.hovered_center = None
        self.closest_centers = []

        # Parameters with default values
        self.threshold_value = 115
        self.threshold_max = 255
        self.kernel_size = 3
        self.max_threshold = 60
        self.min_threshold = 20

        self.flag = False

        # Create UI
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel for controls
        control_panel = QGroupBox()
        control_layout = QVBoxLayout()

        # 设置按钮
        self.load_button = QPushButton("图片加载")
        self.load_button.clicked.connect(self.loadImage)
        control_layout.addWidget(self.load_button)

        # Threshold controls
        threshold_group = QGroupBox("颜色阈值设定")
        threshold_layout = QVBoxLayout()

        # Threshold value slider
        self.threshold_slider = QSlider(Qt.Horizontal)  # 创建一个横向滑块
        self.threshold_slider.setRange(0, 255)  # 设置范围为 0~255
        self.threshold_slider.setValue(self.threshold_value)  # 初始值
        self.threshold_slider.valueChanged.connect(self.updateThreshold)  # 数值变化时执行方法
        threshold_layout.addWidget(QLabel('数值下限：'))  # 文字说明
        threshold_layout.addWidget(self.threshold_slider)  # 加入布局

        # Threshold value spin box
        self.threshold_spin = QSpinBox()  # 创建一个整数输入框
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(self.threshold_value)
        self.threshold_spin.valueChanged.connect(self.thresholdSliderFromSpin)
        threshold_layout.addWidget(self.threshold_spin)

        threshold_group.setLayout(threshold_layout)
        threshold_group.setMaximumSize(300, 150)
        control_layout.addWidget(threshold_group)

        # Morphology controls
        morph_group = QGroupBox("核参数")
        morph_layout = QVBoxLayout()

        # Kernel size slider
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 15)
        self.kernel_slider.setValue(self.kernel_size)
        self.kernel_slider.valueChanged.connect(self.updateMorphology)
        morph_layout.addWidget(QLabel("Kernel Size:"))
        morph_layout.addWidget(self.kernel_slider)

        # Kernel size spin box
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 15)
        self.kernel_spin.setValue(self.kernel_size)
        self.kernel_spin.valueChanged.connect(self.kernelSliderFromSpin)
        morph_layout.addWidget(self.kernel_spin)

        morph_group.setLayout(morph_layout)
        morph_group.setMaximumSize(300, 150)
        control_layout.addWidget(morph_group)

        # Distance controls
        distance_group = QGroupBox("距离参数")
        distance_layout = QVBoxLayout()

        # Max threshold slider
        self.max_threshold_slider = QSlider(Qt.Horizontal)
        self.max_threshold_slider.setRange(1, 200)
        self.max_threshold_slider.setValue(self.max_threshold)
        self.max_threshold_slider.valueChanged.connect(self.updateDistanceParams)
        distance_layout.addWidget(QLabel("最大距离:"))
        distance_layout.addWidget(self.max_threshold_slider)

        # Max threshold spin box
        self.max_threshold_spin = QSpinBox()
        self.max_threshold_spin.setRange(1, 200)
        self.max_threshold_spin.setValue(self.max_threshold)
        self.max_threshold_spin.valueChanged.connect(self.maxThresholdSliderFromSpin)
        distance_layout.addWidget(self.max_threshold_spin)

        # Min threshold slider
        self.min_threshold_slider = QSlider(Qt.Horizontal)
        self.min_threshold_slider.setRange(1, 100)
        self.min_threshold_slider.setValue(self.min_threshold)
        self.min_threshold_slider.valueChanged.connect(self.updateDistanceParams)
        distance_layout.addWidget(QLabel("最小距离:"))
        distance_layout.addWidget(self.min_threshold_slider)

        # Min threshold spin box
        self.min_threshold_spin = QSpinBox()
        self.min_threshold_spin.setRange(1, 100)
        self.min_threshold_spin.setValue(self.min_threshold)
        self.min_threshold_spin.valueChanged.connect(self.minThresholdSliderFromSpin)
        distance_layout.addWidget(self.min_threshold_spin)

        distance_group.setLayout(distance_layout)
        distance_group.setMaximumSize(300, 250)
        control_layout.addWidget(distance_group)

        # 设置按钮
        self.auto_button = QPushButton("自动查找")
        # self.load_button.clicked.connect(self.loadImage)
        control_layout.addWidget(self.auto_button)

        # 设置按钮
        self.save_button = QPushButton("保存图片")
        self.save_button.clicked.connect(self.saveImage)
        control_layout.addWidget(self.save_button)

        control_panel.setLayout(control_layout)

        # Right panel for image display
        image_panel = QVBoxLayout()

        # 创建图像查看器
        self.image_viewer = ImageViewer()
        self.image_viewer.mouseMoved.connect(self.handleMouseMove)
        self.image_viewer.mouseClicked.connect(self.handleMouseClick)

        image_panel.addWidget(self.image_viewer)
        image_widget = QWidget()
        image_widget.setLayout(image_panel)

        # 添加面板到主布局
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(image_widget, stretch=4)

        # 在布局完成后创建比例尺，确保它在图像区域上方
        self.ruler = ScaleRuler(self)
        # 设置比例尺不受布局限制，可以自由移动
        self.ruler.show()

        # 已在上面创建和添加

    def loadImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.processImage()
                self.flag = True

    def processImage(self):
        # Convert to grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, self.threshold_image = cv2.threshold(
            self.gray_image, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Apply morphology
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(self.threshold_image, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(cleaned, kernel, iterations=1)

        # Find contours
        self.contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get centers
        self.centers = []
        self.center_to_contour = {}
        for cnt in self.contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.centers.append((cx, cy))
                self.center_to_contour[(cx, cy)] = cnt

        # Reset tracking variables
        self.used_centers = set()
        self.saved_connections = []
        self.hovered_contour = None
        self.hovered_center = None
        self.closest_centers = []

        self.updateDisplay()

    def updateDisplay(self):
        if self.original_image is None:
            return

        # Create a copy of the original image to draw on
        display_img = self.original_image.copy()

        # Draw all contours (all recognized points)
        for i, cnt in enumerate(self.contours):
            if i == self.hovered_contour:
                # Draw hovered contour in red
                cv2.drawContours(display_img, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

        # Draw all saved connections (persistent display)
        for conn in self.saved_connections:
            if "hull" in conn:
                cv2.polylines(display_img, [conn["hull"]], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw center points of all recognized points
        for center in self.centers:
            cv2.circle(display_img, center, 2, (255, 255, 0), -1)

        # Show mouse hover effects
        if self.hovered_center is not None and self.closest_centers:
            # Mark hovered center point
            cv2.circle(display_img, self.hovered_center, 5, (255, 0, 0), -1)

            # Mark closest center points
            for pt in self.closest_centers:
                cv2.circle(display_img, pt, 5, (255, 255, 255), -1)

            # Connect hovered point with the closest points
            for pt in self.closest_centers:
                cv2.line(display_img, self.hovered_center, pt, (0, 255, 255), 2)

            # Connect the closest points with each other
            for i in range(len(self.closest_centers)):
                for j in range(i + 1, len(self.closest_centers)):
                    cv2.line(display_img, self.closest_centers[i], self.closest_centers[j], (0, 255, 255), 2)

        # Convert BGR to RGB for display
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.image_viewer.setImage(display_img)

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_closest_unused_centers(self, center, k=4):
        """Find the k closest unused centers to the specified center"""
        distances = []
        for c in self.centers:
            if c != center and c not in self.used_centers:
                dist = self.euclidean_distance(center, c)
                if dist < self.max_threshold:
                    distances.append((c, dist))

        distances.sort(key=lambda x: x[1])
        top_k = [c for c, _ in distances[:k]]  # Get the closest k points

        # Check if any points are too close to each other
        merged_group = top_k[:3]  # Default return first 3

        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                if self.euclidean_distance(top_k[i], top_k[j]) < self.min_threshold:
                    merged_group.append(top_k[-1])
                    break

        return merged_group

    def handleMouseMove(self, x, y):
        self.hovered_contour = None
        self.hovered_center = None
        self.closest_centers = []

        for i, cnt in enumerate(self.contours):
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                if cv2.pointPolygonTest(cnt, (x, y), False) >= 0 and center not in self.used_centers:
                    self.hovered_contour = i
                    self.hovered_center = center
                    break

        if self.hovered_center is not None:
            self.closest_centers = self.find_closest_unused_centers(self.hovered_center)

        self.updateDisplay()

    def handleMouseClick(self, x, y):
        if self.hovered_center is not None and len(self.closest_centers) <= 4:
            connection_data = {
                "source": self.hovered_center,
                "connections": self.closest_centers
            }

            # Calculate convex hull for the group
            all_pts = [self.hovered_center] + self.closest_centers
            matched_contours = [self.center_to_contour[pt] for pt in all_pts if pt in self.center_to_contour]
            if matched_contours:
                merged = np.vstack(matched_contours)
                hull = cv2.convexHull(merged)
                connection_data["hull"] = hull

            self.saved_connections.append(connection_data)
            self.used_centers.update([self.hovered_center] + self.closest_centers)
            self.hovered_center = None
            self.closest_centers = []
            self.updateDisplay()

    def updateThreshold(self, value):
        if self.flag:
            self.threshold_value = value
            self.threshold_spin.setValue(value)
            self.processImage()

    def thresholdSliderFromSpin(self, value):
        if self.flag:
            self.threshold_slider.setValue(value)
            self.threshold_value = value
            self.processImage()

    def updateMorphology(self, value):
        if self.flag:
            self.kernel_size = value
            self.kernel_spin.setValue(value)
            self.processImage()

    def kernelSliderFromSpin(self, value):
        if self.flag:
            self.kernel_slider.setValue(value)
            self.kernel_size = value
            self.processImage()

    def updateDistanceParams(self):
        if self.flag:
            self.max_threshold = self.max_threshold_slider.value()
            self.max_threshold_spin.setValue(self.max_threshold)
            self.min_threshold = self.min_threshold_slider.value()
            self.min_threshold_spin.setValue(self.min_threshold)

    def maxThresholdSliderFromSpin(self, value):
        if self.flag:
            self.max_threshold_slider.setValue(value)
            self.max_threshold = value

    def minThresholdSliderFromSpin(self, value):
        if self.flag:
            self.min_threshold_slider.setValue(value)
            self.min_threshold = value

    def saveImage(self):
        if self.flag:
            if self.original_image is None or not self.saved_connections:
                return

            # 创建副本用于绘制
            save_img = self.original_image.copy()

            # 绘制 hull 区域（红色）
            for conn in self.saved_connections:
                hull = conn['hull']
                if hull is not None:
                    cv2.drawContours(save_img, [hull], -1, (0, 0, 255), 2)

            # 在图片底部添加比例尺
            height, width = save_img.shape[:2]

            # 获取当前比例尺的值
            ruler_value = self.ruler.ruler_value

            # 设置比例尺参数
            scale_height = 50  # 比例尺高度
            scale_width = 100  # 比例尺宽度
            margin = 20  # 边距

            # 创建一个新的画布，底部增加比例尺的空间
            new_img = np.zeros((height + scale_height, width, 3), dtype=np.uint8)
            new_img[:height, :] = save_img  # 复制原图到新画布上部
            new_img[height:, :] = 255  # 底部区域设为白色

            # 绘制比例尺（靠右侧显示）
            margin_right = 50  # 右侧边距
            start_x = width - scale_width - margin_right
            end_x = start_x + scale_width
            y_pos = height + scale_height // 2

            # 绘制比例尺主线（增加线条粗细）
            cv2.line(new_img, (start_x, y_pos), (end_x, y_pos), (0, 206, 209), 3)

            # 绘制刻度（增加线条粗细和长度）
            cv2.line(new_img, (start_x, y_pos - 7), (start_x, y_pos + 7), (0, 206, 209), 3)
            cv2.line(new_img, (end_x, y_pos - 7), (end_x, y_pos + 7), (0, 206, 209), 3)

            # 添加文字标注（增加字体大小和粗细）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5  # 增大字体大小
            font_color = (0, 0, 0)
            thickness = 2  # 增加字体粗细

            # 起点标注
            cv2.putText(new_img, "0.0", (start_x - 15, y_pos - 12), font, font_scale, font_color, thickness)

            # 终点标注（显示实际值）
            value_text = f"{ruler_value:.1f} nm"
            cv2.putText(new_img, value_text, (end_x - 50, y_pos - 12), font, font_scale, font_color, thickness)

            # 生成默认文件名带时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"highlight_{timestamp}.png"

            # 弹出保存对话框
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存图片", default_name, "PNG Files (*.png);;JPG Files (*.jpg)"
            )

            if save_path:
                cv2.imwrite(save_path, new_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())