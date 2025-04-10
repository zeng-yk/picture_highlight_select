import sys
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QSpinBox, QGroupBox, QFileDialog, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


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
        self.threshold_slider = QSlider(Qt.Horizontal) # 创建一个横向滑块
        self.threshold_slider.setRange(0, 255) # 设置范围为 0~255
        self.threshold_slider.setValue(self.threshold_value) # 初始值
        self.threshold_slider.valueChanged.connect(self.updateThreshold) # 数值变化时执行方法
        threshold_layout.addWidget(QLabel('数值下限：')) # 文字说明
        threshold_layout.addWidget(self.threshold_slider) # 加入布局


        # Threshold value spin box
        self.threshold_spin = QSpinBox() # 创建一个整数输入框
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(self.threshold_value)
        self.threshold_spin.valueChanged.connect(self.thresholdSliderFromSpin)
        threshold_layout.addWidget(self.threshold_spin)

        threshold_group.setLayout(threshold_layout)
        threshold_group.setMaximumSize(300,150)
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
        morph_group.setMaximumSize(300,150)
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
        self.image_viewer = ImageViewer()
        self.image_viewer.mouseMoved.connect(self.handleMouseMove)
        self.image_viewer.mouseClicked.connect(self.handleMouseClick)

        # Add panels to main layout
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(self.image_viewer, stretch=4)

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

    import time  # 确保在文件顶部引入

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

            # 生成默认文件名带时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"highlight_{timestamp}.png"

            # 弹出保存对话框
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存图片", default_name, "PNG Files (*.png);;JPG Files (*.jpg)"
            )

            if save_path:
                cv2.imwrite(save_path, save_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())