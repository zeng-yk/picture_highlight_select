import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QSpinBox, QGroupBox, QFileDialog, QPushButton, QMessageBox,
                             QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QRectF, QPoint





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

        self.selected_hull_index = None  # 选中的凸包索引
        self.hovered_hull_index = None  # 悬停的凸包索引
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
        # 首先在按钮创建后添加事件连接
        self.auto_button = QPushButton("自动查找")
        self.auto_button.clicked.connect(self.autoFindHulls)
        control_layout.addWidget(self.auto_button)
        # 在界面初始化部分添加删除按钮
        self.delete_button = QPushButton("删除选中凸包")
        self.delete_button.clicked.connect(self.deleteSelectedHull)
        control_layout.addWidget(self.delete_button)

        # 在界面初始化部分添加点数选择控件
        self.points_label = QLabel("每个凸包的点数:")
        control_layout.addWidget(self.points_label)

        self.points_spinner = QSpinBox()
        self.points_spinner.setMinimum(3)  # 凸包至少需要3个点
        self.points_spinner.setMaximum(10)  # 设置一个合理的上限
        self.points_spinner.setValue(4)  # 默认为4个点
        control_layout.addWidget(self.points_spinner)
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

        # 创建原始图像的副本用于绘制
        display_img = self.original_image.copy()

        # 绘制所有轮廓（所有识别到的点）
        for i, cnt in enumerate(self.contours):
            if i == self.hovered_contour:
                # 用红色填充绘制悬停的轮廓
                cv2.drawContours(display_img, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

        # 绘制所有已保存的连接（持久显示）
        for i, conn in enumerate(self.saved_connections):
            if "hull" in conn:
                # 如果是选中的凸包，用紫色高亮显示
                if i == self.selected_hull_index:
                    cv2.polylines(display_img, [conn["hull"]], isClosed=True, color=(255, 0, 255), thickness=3)
                # 如果是悬停的凸包，用黄色高亮显示
                elif hasattr(self, 'hovered_hull_index') and i == self.hovered_hull_index:
                    cv2.polylines(display_img, [conn["hull"]], isClosed=True, color=(0, 255, 255), thickness=2)
                else:
                    cv2.polylines(display_img, [conn["hull"]], isClosed=True, color=(0, 0, 255), thickness=2)

        # 绘制所有识别到的点的中心点
        for center in self.centers:
            cv2.circle(display_img, center, 3, (255, 255, 0), -1)

        # 显示鼠标悬停效果
        if self.hovered_center is not None and self.closest_centers:
            # 标记悬停的中心点
            cv2.circle(display_img, self.hovered_center, 5, (255, 0, 0), -1)

            # 标记最近的中心点
            for pt in self.closest_centers:
                cv2.circle(display_img, pt, 5, (255, 255, 255), -1)

            # 连接悬停点与最近的点
            for pt in self.closest_centers:
                cv2.line(display_img, self.hovered_center, pt, (0, 255, 255), 2)

            # 将最近的点相互连接
            for i in range(len(self.closest_centers)):
                for j in range(i + 1, len(self.closest_centers)):
                    cv2.line(display_img, self.closest_centers[i], self.closest_centers[j], (0, 255, 255), 2)

        # 将BGR转换为RGB用于显示
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.image_viewer.setImage(display_img)
    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    """def find_closest_unused_centers(self, center, k=4):
        #找到指定中心点最近的k个未使用的点，严格返回3个点（加上中心点共4个）
        distances = []
        for c in self.centers:
            if c != center and c not in self.used_centers:
                dist = self.euclidean_distance(center, c)
                if dist < self.max_threshold:
                    distances.append((c, dist))

        distances.sort(key=lambda x: x[1])
        # 直接返回最近的3个点（加上中心点正好4个）
        return [c for c, _ in distances[:3]]  # 严格返回3个最近的点"""

    def find_closest_unused_centers(self, center, k=None):
        """找到指定中心点最近的k-1个未使用的点（加上中心点共k个）"""
        if k is None:
            k = self.points_spinner.value()  # 使用界面上设置的点数

        points_needed = k - 1  # 需要找到的点数（除了中心点）

        distances = []
        for c in self.centers:
            if c != center and c not in self.used_centers:
                dist = self.euclidean_distance(center, c)
                if dist < self.max_threshold:
                    distances.append((c, dist))

        distances.sort(key=lambda x: x[1])
        # 返回最近的points_needed个点
        return [c for c, _ in distances[:points_needed]]

    def handleMouseMove(self, x, y):
        self.hovered_contour = None
        self.hovered_center = None
        self.closest_centers = []

        # 检查鼠标是否悬停在任何未使用的点上
        for i, cnt in enumerate(self.contours):
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                if cv2.pointPolygonTest(cnt, (x, y), False) >= 0 and center not in self.used_centers:
                    self.hovered_contour = i
                    self.hovered_center = center
                    break

        # 只有当找到悬停点时才寻找最近的点
        if self.hovered_center is not None:
            points_per_hull = self.points_spinner.value()
            closest = self.find_closest_unused_centers(self.hovered_center, points_per_hull)
            # 只有当找到正好points_per_hull-1个最近点时才设置
            if len(closest) == points_per_hull - 1:
                self.closest_centers = closest
            else:
                # 如果找不到足够的点，清空悬停状态
                self.hovered_contour = None
                self.hovered_center = None
        # 如果没有悬停在点上，检查是否悬停在凸包上（但不改变选中状态）
        elif self.selected_hull_index is None:  # 只有在没有选中凸包时才检测悬停
            for i, conn in enumerate(self.saved_connections):
                if "hull" in conn:
                    # 检查点是否在凸包内
                    if cv2.pointPolygonTest(conn["hull"], (x, y), False) >= 0:
                        # 这里只是临时高亮显示，不改变选中状态
                        self.hovered_hull_index = i  # 新增一个变量表示悬停的凸包
                        break
                    else:
                        self.hovered_hull_index = None

        self.updateDisplay()

    def handleMouseClick(self, x, y):
        # 如果悬停在点上，处理点的选择和凸包创建
        if self.hovered_center is not None:
            points_per_hull = self.points_spinner.value()
            # 确保有足够的最近点
            if len(self.closest_centers) == points_per_hull - 1:
                connection_data = {
                    "source": self.hovered_center,
                    "connections": self.closest_centers
                }

                # 为这组点计算凸包
                all_pts = [self.hovered_center] + self.closest_centers
                matched_contours = [self.center_to_contour[pt] for pt in all_pts if pt in self.center_to_contour]
                if len(matched_contours) == points_per_hull:  # 确保正好有设定的点数
                    merged = np.vstack(matched_contours)
                    hull = cv2.convexHull(merged)
                    connection_data["hull"] = hull
                    self.saved_connections.append(connection_data)
                    self.used_centers.update([self.hovered_center] + self.closest_centers)

                self.hovered_center = None
                self.closest_centers = []
                self.updateDisplay()
        # 如果点击在凸包上，选中该凸包
        else:
            # 先检查是否点击在任何凸包上
            clicked_on_hull = False
            for i, conn in enumerate(self.saved_connections):
                if "hull" in conn:
                    # 检查点是否在凸包内
                    if cv2.pointPolygonTest(conn["hull"], (x, y), False) >= 0:
                        self.selected_hull_index = i
                        clicked_on_hull = True
                        self.updateDisplay()
                        break

            # 如果点击在空白处，取消选中状态
            if not clicked_on_hull:
                self.selected_hull_index = None
                self.updateDisplay()
    
    def deleteSelectedHull(self):
        """删除选中的凸包"""
        if self.selected_hull_index is not None and 0 <= self.selected_hull_index < len(self.saved_connections):
            # 获取要删除的凸包
            conn = self.saved_connections[self.selected_hull_index]

            # 将凸包中的点标记为未使用
            if "source" in conn:
                self.used_centers.remove(conn["source"])
            if "connections" in conn:
                for pt in conn["connections"]:
                    self.used_centers.remove(pt)

            # 删除凸包
            self.saved_connections.pop(self.selected_hull_index)
            self.selected_hull_index = None

            # 更新显示
            self.updateDisplay()
    """def autoFindHulls(self):
        #自动查找并生成所有可能的四点凸包
        if not self.centers or len(self.centers) < 4:
            QMessageBox.warning(None, "警告", "没有足够的点进行凸包计算")
            return

        # 显示进度对话框
        progress = QProgressDialog("正在自动查找凸包...", "取消", 0, len(self.centers), None)
        progress.setWindowTitle("处理中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # 清除之前的结果
        self.saved_connections = []
        self.used_centers = set()

        # 预先计算所有点之间的距离
        distance_cache = {}
        for i, center1 in enumerate(self.centers):
            for center2 in self.centers:
                if center1 != center2:
                    key = (center1, center2)
                    distance_cache[key] = self.euclidean_distance(center1, center2)

        # 按照点的位置排序，从左上角开始处理
        sorted_centers = sorted(self.centers, key=lambda c: (c[1], c[0]))

        processed = 0

        # 遍历所有未使用的点
        for start_center in sorted_centers:
            if progress.wasCanceled():
                break

            if start_center in self.used_centers:
                continue

            # 更新进度
            progress.setValue(processed)
            processed += 1

            # 使用与handleMouseMove相同的逻辑找到最近的3个点
            closest_centers = self.find_closest_unused_centers(start_center)

            # 确保有足够的最近点（必须是3个，加上中心点共4个）
            if len(closest_centers) == 3:
                connection_data = {
                    "source": start_center,
                    "connections": closest_centers
                }

                # 为这组点计算凸包（严格4个点）
                all_pts = [start_center] + closest_centers
                matched_contours = [self.center_to_contour[pt] for pt in all_pts if pt in self.center_to_contour]
                if len(matched_contours) == 4:  # 确保正好有4个点
                    try:
                        merged = np.vstack(matched_contours)
                        hull = cv2.convexHull(merged)
                        connection_data["hull"] = hull
                        self.saved_connections.append(connection_data)
                        self.used_centers.update([start_center] + closest_centers)
                    except Exception as e:
                        print(f"计算凸包时出错: {e}")

        # 关闭进度对话框
        progress.setValue(len(self.centers))

        # 更新显示
        QApplication.processEvents()  # 确保UI响应
        self.updateDisplay()

        # 显示结果
        QMessageBox.information(None, "完成", f"已找到 {len(self.saved_connections)} 个凸包")"""

    def autoFindHulls(self):
        """自动查找并生成所有可能的多点凸包"""
        if not self.centers or len(self.centers) < 3:
            QMessageBox.warning(None, "警告", "没有足够的点进行凸包计算")
            return

        # 获取用户设置的每个凸包的点数
        points_per_hull = self.points_spinner.value()

        if len(self.centers) < points_per_hull:
            QMessageBox.warning(None, "警告", f"没有足够的点生成{points_per_hull}点凸包")
            return

        # 显示进度对话框
        progress = QProgressDialog(f"正在自动查找{points_per_hull}点凸包...", "取消", 0, len(self.centers), None)
        progress.setWindowTitle("处理中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # 清除之前的结果
        self.saved_connections = []
        self.used_centers = set()

        # 预先计算所有点之间的距离
        distance_cache = {}
        for i, center1 in enumerate(self.centers):
            for center2 in self.centers:
                if center1 != center2:
                    key = (center1, center2)
                    distance_cache[key] = self.euclidean_distance(center1, center2)

        # 按照点的位置排序，从左上角开始处理
        sorted_centers = sorted(self.centers, key=lambda c: (c[1], c[0]))

        processed = 0

        # 遍历所有未使用的点
        for start_center in sorted_centers:
            if progress.wasCanceled():
                break

            if start_center in self.used_centers:
                continue

            # 更新进度
            progress.setValue(processed)
            processed += 1

            # 使用与handleMouseMove相同的逻辑找到最近的points_per_hull-1个点
            closest_centers = self.find_closest_unused_centers(start_center, points_per_hull)

            # 确保有足够的最近点
            if len(closest_centers) == points_per_hull - 1:
                connection_data = {
                    "source": start_center,
                    "connections": closest_centers
                }

                # 为这组点计算凸包
                all_pts = [start_center] + closest_centers
                matched_contours = [self.center_to_contour[pt] for pt in all_pts if pt in self.center_to_contour]
                if len(matched_contours) == points_per_hull:  # 确保正好有设定的点数
                    try:
                        merged = np.vstack(matched_contours)
                        hull = cv2.convexHull(merged)
                        connection_data["hull"] = hull
                        self.saved_connections.append(connection_data)
                        self.used_centers.update([start_center] + closest_centers)
                    except Exception as e:
                        print(f"计算凸包时出错: {e}")

        # 关闭进度对话框
        progress.setValue(len(self.centers))

        # 更新显示
        QApplication.processEvents()  # 确保UI响应
        self.updateDisplay()

        # 显示结果
        QMessageBox.information(None, "完成", f"已找到 {len(self.saved_connections)} 个{points_per_hull}点凸包")
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
            font_scale = 0.5  # 字体大小
            font_color = (0, 0, 0)
            thickness = 2  # 字体粗细

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