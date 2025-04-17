import sys # 系统相关功能模块
import time # 时间相关功能模块

import cv2 # OpenCV图像处理库
import numpy as np # 数值计算库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QSpinBox, QGroupBox, QFileDialog, QPushButton, QCheckBox)  # PyQt5界面组件
from PyQt5.QtCore import Qt, pyqtSignal  # Qt核心功能和信号机制
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor  # Qt图形界面相关组件
from PyQt5.QtWidgets import QWidget, QInputDialog  # Qt部件和对话框
from PyQt5.QtGui import QPainter, QPen, QFont  # Qt绘图相关组件
from PyQt5.QtCore import Qt, QRectF, QPoint  # Qt核心功能和几何图形


class ScaleRuler(QWidget):
    """
    比例尺控件类
    用于在图像上显示可调整的比例尺，支持拖动调整长度和位置
    可通过拖动端点改变比例尺长度，通过拖动中间部分移动整个比例尺
    右键点击可设置比例尺对应的真实值
    """
    def __init__(self, parent=None):
        """
        初始化比例尺控件

        参数:
            parent: 父窗口对象
        """
        super().__init__(parent)
        self.height_ = 40 # 控件高度，单位为像素
        self.setFixedHeight(self.height_) # 设置控件固定高度
        self.ruler_start = 10 # 比例尺起点位置，单位为像素
        self.ruler_end = 150  # 初始比例尺长度（像素）
        self.max_display_length = 800  # 增加最大显示长度，允许更长的比例尺
        self.dragging = False # 标记是否正在拖动右侧端点
        self.dragging_start = False  # 标记是否正在拖动左侧端点
        self.ruler_value = 1.0  # 比例尺代表的真实长度值（用户定义，单位为nm）
        self.setMouseTracking(True) # 启用鼠标跟踪，即使不按下鼠标按钮也能接收鼠标移动事件
        self.setCursor(Qt.OpenHandCursor) # 设置光标样式为手型，提示用户可以拖动
        # 允许整个比例尺控件在窗口中自由移动
        self.ruler_dragging = False # 标记是否正在拖动整个比例尺
        self.drag_start_pos = None # 拖动起始位置，用于计算移动距离
        # 设置初始位置在图片区域中间位置
        self.setGeometry(300, 50, 500, self.height_)  # 更宽的初始宽度
        self.setMinimumWidth(200)  # 设置最小宽度
        self.setMaximumWidth(2000)  # 设置最大宽度，允许更长的比例尺

    def paintEvent(self, event):
        """
        绘制比例尺控件

        参数:
            event: 绘制事件对象，包含绘制相关信息
        """
        # 使用双缓冲绘制减少闪烁
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)  # 启用抗锯齿
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)  # 平滑像素图变换

        # 设置画笔
        pen = QPen(QColor(0, 206, 209), 2)  # 创建青色画笔，线宽为2像素，更易于观察
        painter.setPen(pen)  # 设置画笔到绘图对象

        # 绘制比例尺主线（水平线）
        y = self.height_ // 2  # 计算垂直中心位置，确保比例尺在控件中间
        painter.drawLine(self.ruler_start, y, self.ruler_end, y)  # 从起点到终点绘制水平线

        # 绘制刻度，限制最多显示10个分度值
        ruler_length = self.ruler_end - self.ruler_start # 计算比例尺长度（像素）

        # 确保至少有两个刻度点，最多10个
        if ruler_length > 20:  # 确保有足够的空间显示刻度
            # 限制最多显示10个刻度
            max_ticks = 10  # 最大刻度数
            # 根据比例尺长度计算合适的刻度数量，确保至少有2个刻度
            num_ticks = min(max_ticks, max(2, int(ruler_length / 30) + 1))  # 每30像素放置一个刻度
            # 计算每个刻度的位置，均匀分布
            tick_positions = [self.ruler_start + i * (ruler_length / (num_ticks - 1)) for i in range(num_ticks)]

            # 设置字体一次，避免重复设置
            painter.setFont(QFont('Arial', 8))

            # 绘制每个刻度及其对应的值
            for i, x in enumerate(tick_positions):
                x_pos = int(x)  # 转换为整数坐标，避免浮点数绘制问题
                painter.drawLine(x_pos, y - 5, x_pos, y + 5)  # 绘制垂直刻度线，向上和向下各5像素
                # 计算每个刻度对应的实际值（按比例）
                value = (i / (num_ticks - 1)) * self.ruler_value  # 根据位置比例计算实际值
                painter.drawText(QPoint(x_pos - 10, y - 8), f"{value:.1f}")  # 在刻度上方绘制数值，保留一位小数
        else:
            # 如果比例尺太短（<=20像素），至少显示起点和终点两个刻度
            painter.setFont(QFont('Arial', 8))  # 设置小字体
            painter.drawLine(self.ruler_start, y - 5, self.ruler_start, y + 5)  # 绘制起点刻度线
            painter.drawLine(self.ruler_end, y - 5, self.ruler_end, y + 5)  # 绘制终点刻度线
            painter.drawText(QPoint(self.ruler_start - 10, y - 8), "0.0")  # 绘制起点刻度值（0.0）
            painter.drawText(QPoint(self.ruler_end - 10, y - 8), f"{self.ruler_value:.1f}")  # 绘制终点刻度值

        # 绘制总长值
        painter.drawText(QPoint(self.ruler_start, y + 20), f"{self.ruler_value:.1f} nm")

    def mousePressEvent(self, event):
        """
        鼠标按下事件处理

        参数:
            event: 鼠标事件对象，包含鼠标位置和按键信息
        """
        if abs(event.x() - self.ruler_end) < 10:
            # 如果鼠标点击位置在右侧端点附近（10像素范围内），开始右侧端点拖动
            self.dragging = True  # 设置右侧端点拖动标志
            self.setCursor(Qt.ClosedHandCursor)  # 改变鼠标光标为抓取状态
        elif abs(event.x() - self.ruler_start) < 10:
            # 如果鼠标点击位置在左侧端点附近（10像素范围内），开始左侧端点拖动
            self.dragging_start = True  # 设置左侧端点拖动标志
            self.setCursor(Qt.ClosedHandCursor)  # 改变鼠标光标为抓取状态
        elif self.ruler_start <= event.x() <= self.ruler_end:
            # 如果点击在比例尺线上，但不是端点，则开始整个比例尺的拖动
            if event.button() == Qt.RightButton:
                # 右键点击弹出设置真实长度对话框
                val, ok = QInputDialog.getDouble(self, "设置比例尺值", "当前比例尺对应的真实值：",
                                                 value=self.ruler_value, min=0.01)  # 显示对话框，最小值为0.01
                if ok:  # 如果用户点击了确定
                    self.ruler_value = val  # 更新比例尺真实值
                    self.update()  # 重绘比例尺
            else:
                # 左键点击开始拖动整个比例尺
                self.ruler_dragging = True  # 设置整体拖动标志
                self.drag_start_pos = event.pos()  # 记录拖动起始位置
                self.setCursor(Qt.ClosedHandCursor)  # 改变鼠标光标为抓取状态

    def mouseMoveEvent(self, event):
        """
        鼠标移动事件处理

        参数:
            event: 鼠标事件对象，包含鼠标当前位置信息
        """
        if self.dragging:
            # 拖动右侧端点
            # 允许数值随鼠标移动而变化，不限制显示长度
            raw_end = max(self.ruler_start + 10, event.x())

            # 直接更新结束位置，不限制最大长度
            self.ruler_end = raw_end

            # 如果比例尺长度超过控件宽度，自动调整控件宽度
            ruler_length = self.ruler_end - self.ruler_start  # 计算当前比例尺长度
            if ruler_length + 40 > self.width():  # 增加边距，确保有足够空间
                new_width = ruler_length + 40  # 左右各预留20像素
                self.resize(new_width, self.height_)  # 直接调整控件大小

            self.update()
        elif self.dragging_start:
            # 拖动左侧端点，只允许向右移动，不能向左移动超过初始位置
            new_start = event.x()
            # 确保不能向左拖动超过初始位置，且与右端点保持最小距离
            if 10 <= new_start < self.ruler_end - 10:  # 10是初始位置，不能向左拖动
                self.ruler_start = new_start  # 更新比例尺起点位置
                self.update()  # 重绘控件
        elif self.ruler_dragging:
            # 拖动整个比例尺逻辑
            delta = event.pos() - self.drag_start_pos  # 计算鼠标移动距离
            self.move(self.pos() + delta)  # 移动整个控件
            self.drag_start_pos = event.pos()  # 更新拖动起始位置为当前位置
            # 防止闪烁，使用update()而不是repaint()
            self.update() # 重绘

    def mouseReleaseEvent(self, event):
        """
        鼠标释放事件处理
        当用户释放鼠标按钮时调用，用于结束拖动操作
        参数:
            event: 鼠标事件对象，包含鼠标释放时的信息
        """
        self.dragging = False  # 重置右侧端点拖动标记，结束右侧端点拖动
        self.dragging_start = False  # 重置左侧端点拖动标记，结束左侧端点拖动
        self.ruler_dragging = False  # 重置整体拖动标记，结束整体拖动
        self.drag_start_pos = None  # 清除拖动起始位置，释放引用
        self.setCursor(Qt.OpenHandCursor)  # 恢复光标样式为手型，提示可以继续拖动


class ImageViewer(QWidget):
    """
    图像查看器类
    用于显示图像并处理鼠标事件
    提供鼠标移动和点击的坐标信息，支持图像缩放和居中显示
    """
    mouseMoved = pyqtSignal(int, int)  # 鼠标移动信号，传递图像坐标（x, y）
    mouseClicked = pyqtSignal(int, int)  # 鼠标点击信号，传递图像坐标（x, y）

    def __init__(self):
        """
        初始化图像查看器
        设置基本属性并启用鼠标跟踪
        """
        super().__init__()  # 调用父类初始化方法
        self.image = None  # 原始图像，OpenCV格式（BGR）
        self.display_image = None  # 显示用的图像副本，用于绘制
        self.setMouseTracking(True)  # 启用鼠标跟踪，即使不按下鼠标按钮也能接收鼠标移动事件

    def setImage(self, image):
        """
        设置要显示的图像
        用于更新图像查看器中显示的图像内容

        参数:
            image: 要显示的图像（OpenCV格式，BGR色彩空间）
        """
        self.image = image  # 保存原始图像引用
        self.display_image = image.copy()  # 创建显示用的图像副本，避免修改原始图像
        self.update()  # 触发重绘，更新显示内容

    def paintEvent(self, event):
        """
        绘制图像
        当控件需要重绘时自动调用此方法

        参数:
            event: 绘制事件对象，包含重绘相关信息
        """
        if self.display_image is not None:  # 确保有图像可以显示
            painter = QPainter(self)  # 创建绘图对象

            # 将OpenCV图像（BGR格式）转换为QImage（RGB格式）
            height, width, channel = self.display_image.shape  # 获取图像尺寸和通道数
            bytesPerLine = 3 * width  # 计算每行字节数（3通道，每通道1字节）
            qImg = QImage(self.display_image.data, width, height, bytesPerLine, QImage.Format_RGB888)  # 创建QImage

            # 缩放图像以适应控件大小，同时保持纵横比，使用平滑变换提高质量
            scaled_img = qImg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 计算位置以使图像在控件中居中显示
            x = (self.width() - scaled_img.width()) // 2  # 计算水平居中位置
            y = (self.height() - scaled_img.height()) // 2  # 计算垂直居中位置
            painter.drawImage(x, y, scaled_img)  # 在计算的位置绘制缩放后的图像

    def mouseMoveEvent(self, event):
        """
        鼠标移动事件处理
        当鼠标在控件上移动时调用，将控件坐标转换为图像坐标并发送信号

        参数:
            event: 鼠标事件对象，包含鼠标位置信息
        """
        if self.display_image is not None:  # 确保有图像可以处理
            # 计算实际图像位置和大小
            img_height, img_width = self.display_image.shape[:2]  # 获取图像高度和宽度
            widget_width = self.width()  # 获取控件宽度
            widget_height = self.height()  # 获取控件高度

            # 计算缩放因子和偏移量，用于坐标转换
            scale = min(widget_width / img_width, widget_height / img_height)  # 计算缩放比例（取较小值保持纵横比）
            scaled_width = int(img_width * scale)  # 计算缩放后的宽度
            scaled_height = int(img_height * scale)  # 计算缩放后的高度
            x_offset = (widget_width - scaled_width) // 2  # 计算X轴偏移量（居中显示）
            y_offset = (widget_height - scaled_height) // 2  # 计算Y轴偏移量（居中显示）

            # 将控件坐标转换为原始图像坐标
            img_x = int((event.x() - x_offset) / scale)  # 计算图像X坐标
            img_y = int((event.y() - y_offset) / scale)  # 计算图像Y坐标

            # 仅当鼠标位置在图像内部时才发出信号
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                self.mouseMoved.emit(img_x, img_y)  # 发送鼠标移动信号，传递图像坐标


    def mousePressEvent(self, event):
        """
        鼠标按下事件处理
        当鼠标按钮在控件上按下时调用，将控件坐标转换为图像坐标并发送点击信号

        参数:
            event: 鼠标事件对象，包含鼠标位置和按键信息
        """
        # 只处理左键点击且有图像显示的情况
        if event.button() == Qt.LeftButton and self.display_image is not None:
            # 与mouseMoveEvent中相同的坐标转换逻辑
            img_height, img_width = self.display_image.shape[:2]  # 获取图像高度和宽度
            widget_width = self.width()  # 获取控件宽度
            widget_height = self.height()  # 获取控件高度

            # 计算缩放比例和偏移量
            scale = min(widget_width / img_width, widget_height / img_height)  # 计算缩放比例
            scaled_width = int(img_width * scale)  # 计算缩放后的宽度
            scaled_height = int(img_height * scale)  # 计算缩放后的高度
            x_offset = (widget_width - scaled_width) // 2  # 计算X轴偏移量
            y_offset = (widget_height - scaled_height) // 2  # 计算Y轴偏移量

            # 将控件坐标转换为原始图像坐标
            img_x = int((event.x() - x_offset) / scale)  # 计算图像X坐标
            img_y = int((event.y() - y_offset) / scale)  # 计算图像Y坐标

            # 确保点击位置在图像范围内
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                self.mouseClicked.emit(img_x, img_y)  # 发送鼠标点击信号，传递图像坐标


class ImageProcessor(QMainWindow):
    """
    图像处理主窗口类
    用于亮点提取和处理
    提供图像加载、处理、显示和交互功能，支持亮点检测和连接
    """
    def __init__(self):
        """
        初始化图像处理器
        设置窗口属性、初始化变量和创建用户界面
        """
        super().__init__()  # 调用父类初始化方法
        self.setWindowTitle("亮点提取工具")  # 设置窗口标题
        self.setGeometry(100, 100, 1200, 800)  # 设置窗口大小和位置（x, y, 宽度, 高度）

        # 初始化图像相关变量
        self.original_image = None  # 原始图像，OpenCV格式（BGR）
        self.gray_image = None  # 灰度图像，单通道
        self.threshold_image = None  # 阈值处理后的二值图像
        self.processed_image = None  # 处理后的图像，用于显示
        self.contours = []  # 检测到的轮廓列表
        self.centers = []  # 轮廓中心点列表
        self.center_to_contour = {}  # 中心点到轮廓的映射字典
        self.used_centers = set()  # 已使用的中心点集合，用于记录已经处理过的点
        self.saved_connections = []  # 保存的连接列表，存储已确认的点之间的连接

        # 交互相关变量
        self.hovered_contour = None  # 当前鼠标悬停的轮廓索引
        self.hovered_center = None  # 当前鼠标悬停的中心点坐标
        self.closest_centers = []  # 与当前悬停点最近的中心点列表

        # 图像处理参数默认值
        self.threshold_value = 115  # 二值化阈值，低于此值的像素被设为0
        self.threshold_max = 255  # 二值化最大值，高于阈值的像素被设为此值
        self.kernel_size = 3  # 形态学操作的核大小，影响噪点去除效果
        self.max_threshold = 60  # 最大距离阈值，超过此距离的点不会被连接
        self.min_threshold = 20  # 最小距离阈值，小于此距离的点会被优先连接

        self.flag = False # 标记是否已加载图像

        # 创建 UI
        self.initUI()

    def initUI(self):
        """
        初始化用户界面
        """
        # 主窗口和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QGroupBox()
        control_layout = QVBoxLayout()

        # 设置图片按钮
        self.load_button = QPushButton("图片加载")
        self.load_button.clicked.connect(self.loadImage)
        control_layout.addWidget(self.load_button)

        # 阈值控制组
        threshold_group = QGroupBox("颜色阈值设定")
        threshold_layout = QVBoxLayout()

        # 阈值滑块
        self.threshold_slider = QSlider(Qt.Horizontal)  # 创建一个横向滑块
        self.threshold_slider.setRange(0, 255)  # 设置范围为 0~255
        self.threshold_slider.setValue(self.threshold_value)  # 初始值
        self.threshold_slider.valueChanged.connect(self.updateThreshold)  # 数值变化时执行方法
        threshold_layout.addWidget(QLabel('数值下限：'))  # 文字说明
        threshold_layout.addWidget(self.threshold_slider)  # 加入布局

        # 阈值数值输入框
        self.threshold_spin = QSpinBox()  # 创建一个整数输入框
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(self.threshold_value)
        self.threshold_spin.valueChanged.connect(self.thresholdSliderFromSpin)
        threshold_layout.addWidget(self.threshold_spin)

        threshold_group.setLayout(threshold_layout)
        threshold_group.setMaximumSize(300, 150)
        control_layout.addWidget(threshold_group)

        # 形态学控制组
        morph_group = QGroupBox("核参数")
        morph_layout = QVBoxLayout()

        # 核大小滑块
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 15)
        self.kernel_slider.setValue(self.kernel_size)
        self.kernel_slider.valueChanged.connect(self.updateMorphology)
        morph_layout.addWidget(QLabel("Kernel Size:"))
        morph_layout.addWidget(self.kernel_slider)

        # 核大小数值输入框
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 15)
        self.kernel_spin.setValue(self.kernel_size)
        self.kernel_spin.valueChanged.connect(self.kernelSliderFromSpin)
        morph_layout.addWidget(self.kernel_spin)

        morph_group.setLayout(morph_layout)
        morph_group.setMaximumSize(300, 150)
        control_layout.addWidget(morph_group)

        # 冗余搜索
        # 创建复选框
        check_panel = QGroupBox()
        checkbox = QCheckBox("冗余搜索", check_panel)

        # 常用方法
        checkbox.setChecked(False)  # 设置默认选中
        checkbox.setText("冗余搜索")  # 更改文本
        check_panel.setMaximumSize(300, 20)
        control_layout.addWidget(check_panel)
        # # 信号连接
        # checkbox.stateChanged.connect(lambda state: print("状态:", state))

        # 距离参数控制组
        distance_group = QGroupBox("距离参数")
        distance_layout = QVBoxLayout()

        # 最大距离滑块
        self.max_threshold_slider = QSlider(Qt.Horizontal)
        self.max_threshold_slider.setRange(1, 200)
        self.max_threshold_slider.setValue(self.max_threshold)
        self.max_threshold_slider.valueChanged.connect(self.updateDistanceParams)
        distance_layout.addWidget(QLabel("最大距离:"))
        distance_layout.addWidget(self.max_threshold_slider)

        # 最大距离数值输入框
        self.max_threshold_spin = QSpinBox()
        self.max_threshold_spin.setRange(1, 200)
        self.max_threshold_spin.setValue(self.max_threshold)
        self.max_threshold_spin.valueChanged.connect(self.maxThresholdSliderFromSpin)
        distance_layout.addWidget(self.max_threshold_spin)

        # 最小距离滑块
        self.min_threshold_slider = QSlider(Qt.Horizontal)
        self.min_threshold_slider.setRange(1, 100)
        self.min_threshold_slider.setValue(self.min_threshold)
        self.min_threshold_slider.valueChanged.connect(self.updateDistanceParams)
        distance_layout.addWidget(QLabel("最小距离:"))
        distance_layout.addWidget(self.min_threshold_slider)

        # 最小距离数值输入框
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

        # 右侧图像显示面板
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
        """
        加载图像文件
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.processImage()  # 处理图像
                self.flag = True  # 设置已加载图像标记

    def processImage(self):
        """
        处理图像，包括灰度转换、阈值处理、形态学操作和轮廓检测
        """
        # 转换为灰度图
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # 应用阈值
        _, self.threshold_image = cv2.threshold(
            self.gray_image, self.threshold_value, 255, cv2.THRESH_BINARY)

        # 应用形态学操作
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(self.threshold_image, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(cleaned, kernel, iterations=1)

        # 查找轮廓
        self.contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 获取中心点
        self.centers = []
        self.center_to_contour = {}
        for cnt in self.contours:
            M = cv2.moments(cnt)  # 计算轮廓矩
            if M['m00'] > 0:  # 确保面积大于0
                cx = int(M['m10'] / M['m00'])  # 计算x坐标
                cy = int(M['m01'] / M['m00'])  # 计算y坐标
                self.centers.append((cx, cy))  # 添加中心点
                self.center_to_contour[(cx, cy)] = cnt  # 建立中心点到轮廓的映射

        # 重置跟踪变量
        self.used_centers = set()  # 清空已使用中心点
        self.saved_connections = []  # 清空保存的连接
        self.hovered_contour = None  # 清空悬停轮廓
        self.hovered_center = None  # 清空悬停中心点
        self.closest_centers = []  # 清空最近中心点

        self.updateDisplay()  # 更新显示

    def updateDisplay(self):
        """
        更新图像显示
        """
        if self.original_image is None:
            return

        # 创建原始图像的副本用于绘制
        display_img = self.original_image.copy()  # display_img：用于显示的图像副本

        # 绘制所有轮廓（所有识别的点）
        for i, cnt in enumerate(self.contours):  # i：轮廓索引，cnt：轮廓点集
            if i == self.hovered_contour:
                # 绘制当前鼠标悬停的轮廓（红色填充）
                cv2.drawContours(display_img, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)

        # 绘制所有已保存的连接（持久显示）
        for conn in self.saved_connections:  # conn：连接信息字典
            if "hull" in conn:
                # 绘制多边形轮廓线（红色）
                cv2.polylines(display_img, [conn["hull"]], isClosed=True, color=(0, 0, 255), thickness=2)

        # 绘制所有识别点的中心点
        for center in self.centers:  # center：中心点坐标(x, y)
            cv2.circle(display_img, center, 2, (255, 255, 0), -1)  # 黄色小圆点

        # 显示鼠标悬停效果
        if self.hovered_center is not None and self.closest_centers:
            # 标记悬停的中心点（蓝色大圆点）
            cv2.circle(display_img, self.hovered_center, 5, (255, 0, 0), -1)

            # 标记与悬停点最近的中心点（白色大圆点）
            for pt in self.closest_centers: # pt：最近中心点坐标
                cv2.circle(display_img, pt, 5, (255, 255, 255), -1)

            # 用线连接悬停点与最近的中心点（黄色线）
            for pt in self.closest_centers:
                cv2.line(display_img, self.hovered_center, pt, (0, 255, 255), 2)

            # 将最近的中心点两两相连（黄色线）
            for i in range(len(self.closest_centers)):
                for j in range(i + 1, len(self.closest_centers)):
                    cv2.line(display_img, self.closest_centers[i], self.closest_centers[j], (0, 255, 255), 2)

        # 将BGR格式转换为RGB格式以便显示
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.image_viewer.setImage(display_img) # 更新显示控件中的图像

    def euclidean_distance(self, p1, p2):
        """
        计算两点之间的欧氏距离
        参数:
            p1: 第一个点的坐标，格式为(x, y)
            p2: 第二个点的坐标，格式为(x, y)
        返回:
            两点之间的欧氏距离
        """
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def find_closest_unused_centers(self, center, k=4):
        """
        查找距离指定中心点最近的k个未被使用的中心点
        参数:
            center: 当前中心点坐标，格式为(x, y)
            k: 需要查找的最近点数量，默认为4
        返回:
            merged_group: 最近的中心点列表
        """
        distances = []  # 存储每个未使用点与当前点的距离
        for c in self.centers:
            if c != center and c not in self.used_centers:
                dist = self.euclidean_distance(center, c)  # 计算距离
                if dist < self.max_threshold:
                    distances.append((c, dist))  # 添加到距离列表

        distances.sort(key=lambda x: x[1])  # 按距离升序排序
        top_k = [c for c, _ in distances[:k]]  # 取最近的k个点

        # 检查最近点之间是否有距离过近的情况
        merged_group = top_k[:3]  # 默认返回前三个

        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                if self.euclidean_distance(top_k[i], top_k[j]) < self.min_threshold:
                    merged_group.append(top_k[-1])  # 如果有距离过近的点，加入最后一个点
                    break

        return merged_group

    def handleMouseMove(self, x, y):
        """
        鼠标移动事件处理函数
        根据鼠标位置判断是否悬停在某个轮廓上，并查找最近的中心点
        参数:
            x: 鼠标在图像中的x坐标
            y: 鼠标在图像中的y坐标
        """
        self.hovered_contour = None  # 当前悬停的轮廓索引
        self.hovered_center = None   # 当前悬停的中心点坐标
        self.closest_centers = []    # 最近的中心点列表

        for i, cnt in enumerate(self.contours):
            M = cv2.moments(cnt)  # 计算轮廓的矩
            if M['m00'] > 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))  # 计算中心点
                if cv2.pointPolygonTest(cnt, (x, y), False) >= 0 and center not in self.used_centers:
                    self.hovered_contour = i  # 记录悬停的轮廓索引
                    self.hovered_center = center  # 记录悬停的中心点
                    break

        if self.hovered_center is not None:
            self.closest_centers = self.find_closest_unused_centers(self.hovered_center)  # 查找最近的中心点

        self.updateDisplay() # 更新显示

    def handleMouseClick(self, x, y):
        """
        鼠标点击事件处理函数
        当点击悬停中心点时，保存该点与最近点的连接关系，并计算凸包
        参数:
            x: 鼠标在图像中的x坐标
            y: 鼠标在图像中的y坐标
        """
        if self.hovered_center is not None and len(self.closest_centers) <= 4:
            connection_data = {
                "source": self.hovered_center,  # 当前中心点
                "connections": self.closest_centers  # 最近的中心点列表
            }

            # 计算该组点的凸包
            all_pts = [self.hovered_center] + self.closest_centers  # 所有相关点
            matched_contours = [self.center_to_contour[pt] for pt in all_pts if pt in self.center_to_contour]  # 匹配的轮廓
            if matched_contours:
                merged = np.vstack(matched_contours)  # 合并所有轮廓点
                hull = cv2.convexHull(merged)  # 计算凸包
                connection_data["hull"] = hull  # 保存凸包

            self.saved_connections.append(connection_data)  # 保存连接关系
            self.used_centers.update([self.hovered_center] + self.closest_centers)  # 标记已使用的中心点
            self.hovered_center = None  # 重置悬停中心点
            self.closest_centers = []   # 重置最近中心点
            self.updateDisplay()  # 更新显示

    def updateThreshold(self, value):
        """
        更新阈值滑块时的处理函数
        参数:
            value: 新的阈值
        """
        if self.flag:
            self.threshold_value = value  # 更新阈值
            self.threshold_spin.setValue(value)  # 同步数值输入框
            self.processImage()  # 重新处理图像

    def thresholdSliderFromSpin(self, value):
        """
        阈值数值输入框变化时的处理函数
        参数:
            value: 新的阈值
        """
        if self.flag:
            self.threshold_slider.setValue(value)  # 同步滑块
            self.threshold_value = value  # 更新阈值
            self.processImage()  # 重新处理图像

    def updateMorphology(self, value):
        """
        更新形态学核大小滑块时的处理函数
        参数:
            value: 新的核大小
        """
        if self.flag:
            self.kernel_size = value  # 更新核大小
            self.kernel_spin.setValue(value)  # 同步数值输入框
            self.processImage()  # 重新处理图像

    def kernelSliderFromSpin(self, value):
        """
        形态学核大小数值输入框变化时的处理函数
        参数:
            value: 新的核大小
        """
        if self.flag:
            self.kernel_slider.setValue(value)  # 同步滑块
            self.kernel_size = value  # 更新核大小
            self.processImage()  # 重新处理图像

    def updateDistanceParams(self):
        """
        更新距离参数（最大/最小距离）
        """
        if self.flag:
            self.max_threshold = self.max_threshold_slider.value()  # 更新最大距离
            self.max_threshold_spin.setValue(self.max_threshold)    # 同步数值输入框
            self.min_threshold = self.min_threshold_slider.value()  # 更新最小距离
            self.min_threshold_spin.setValue(self.min_threshold)    # 同步数值输入框

    def maxThresholdSliderFromSpin(self, value):
        """
        最大距离数值输入框变化时的处理函数
        参数:
            value: 新的最大距离
        """
        if self.flag:
            self.max_threshold_slider.setValue(value)  # 同步滑块
            self.max_threshold = value  # 更新最大距离

    def minThresholdSliderFromSpin(self, value):
        """
        最小距离数值输入框变化时的处理函数
        参数:
            value: 新的最小距离
        """
        if self.flag:
            self.min_threshold_slider.setValue(value)  # 同步滑块
            self.min_threshold = value  # 更新最小距离

    # def auto_find(self):

    def saveImage(self):
        """
        保存当前处理结果为图片文件，并在底部添加比例尺
        """
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