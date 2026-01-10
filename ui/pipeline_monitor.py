# -*- coding: utf-8 -*-
"""
流水线监控可视化组件
实时显示图像转换、数据队列、GPU推理、CPU推理四个管线的工作负载状态
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from ui.styles import COLORS


class PipelineMonitorWidget(QWidget):
    """流水线监控可视化组件"""
    
    # 默认绘制区域限制（像素）
    DEFAULT_PIPELINE_WIDTH = 180  # 每个管线的默认宽度
    DEFAULT_PIPELINE_HEIGHT = 80  # 每个管线的默认高度
    DEFAULT_JOB_HEIGHT = 10  # 每个job矩形的默认高度
    DEFAULT_JOB_SPACING = 3  # job之间的间距
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)
        
        # 流水线状态数据
        self.pipeline_data = {
            'conversion': {
                'workers': 0,
                'active_jobs': [],
            },
            'queue': {
                'size': 0,
                'max_size': 100,
            },
            'inference_gpu': {
                'workers': 0,
                'active_jobs': [],
            },
            'inference_cpu': {
                'workers': 0,
                'active_jobs': [],
            }
        }
        
        # 脉冲动画数据
        self.pulse_animations = {
            'conversion_to_queue': 0.0,
            'queue_to_gpu': 0.0,
            'queue_to_cpu': 0.0,
        }
        
        # 启动定时器更新动画
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animations)
        self.animation_timer.start(50)  # 每50ms更新一次
        
        # 脉冲速度
        self.pulse_speed = 0.1
    
    def _update_animations(self):
        """更新脉冲动画"""
        # 检查是否有数据流动
        has_flow_1 = (self.pipeline_data['conversion']['workers'] > 0 and 
                      any(self.pipeline_data['conversion']['active_jobs']))
        has_flow_2_gpu = (self.pipeline_data['queue']['size'] > 0 and 
                         self.pipeline_data['inference_gpu']['workers'] > 0)
        has_flow_2_cpu = (self.pipeline_data['queue']['size'] > 0 and 
                         self.pipeline_data['inference_cpu']['workers'] > 0)
        
        # 更新脉冲动画
        if has_flow_1:
            self.pulse_animations['conversion_to_queue'] += self.pulse_speed
            if self.pulse_animations['conversion_to_queue'] > 1.0:
                self.pulse_animations['conversion_to_queue'] = 0.0
        else:
            self.pulse_animations['conversion_to_queue'] = 0.0
            
        if has_flow_2_gpu:
            self.pulse_animations['queue_to_gpu'] += self.pulse_speed
            if self.pulse_animations['queue_to_gpu'] > 1.0:
                self.pulse_animations['queue_to_gpu'] = 0.0
        else:
            self.pulse_animations['queue_to_gpu'] = 0.0
            
        if has_flow_2_cpu:
            self.pulse_animations['queue_to_cpu'] += self.pulse_speed
            if self.pulse_animations['queue_to_cpu'] > 1.0:
                self.pulse_animations['queue_to_cpu'] = 0.0
        else:
            self.pulse_animations['queue_to_cpu'] = 0.0
        
        self.update()  # 触发重绘
    
    def update_pipeline_data(self, data):
        """更新流水线数据"""
        if 'conversion' in data:
            self.pipeline_data['conversion'].update(data['conversion'])
        if 'queue' in data:
            self.pipeline_data['queue'].update(data['queue'])
        if 'inference_gpu' in data:
            self.pipeline_data['inference_gpu'].update(data['inference_gpu'])
        if 'inference_cpu' in data:
            self.pipeline_data['inference_cpu'].update(data['inference_cpu'])
        self.update()
    
    def paintEvent(self, event):
        """绘制流水线可视化"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # 计算三个列的位置（图像转换、数据队列、推理区域）
        section_width = width / 3
        center_y = height / 2
        
        # 计算缩放因子，使绘制区域适应默认大小
        scale_x = min(1.0, width / (self.DEFAULT_PIPELINE_WIDTH * 3))
        scale_y = min(1.0, (height - 40) / (self.DEFAULT_PIPELINE_HEIGHT * 2))  # 推理区域需要容纳GPU和CPU上下排列
        
        # 管线宽度和高度（根据缩放因子调整）
        pipeline_width = self.DEFAULT_PIPELINE_WIDTH * scale_x
        pipeline_height = self.DEFAULT_PIPELINE_HEIGHT * scale_y
        job_height = self.DEFAULT_JOB_HEIGHT * scale_y
        job_spacing = self.DEFAULT_JOB_SPACING * scale_y
        
        # 绘制三个列
        # 1. 图像转换管线（第一列）
        conversion_x = section_width * 0.5 - pipeline_width / 2
        self._draw_pipeline(
            painter, 
            conversion_x, 
            center_y, 
            pipeline_width,
            pipeline_height,
            self.pipeline_data['conversion'],
            job_height,
            job_spacing,
            "图像转换"
        )
        
        # 2. 数据队列管线（第二列）
        queue_x = section_width * 1.5 - pipeline_width / 2
        self._draw_queue(
            painter,
            queue_x,
            center_y,
            pipeline_width,
            pipeline_height,
            self.pipeline_data['queue'],
            job_height,
            "数据队列"
        )
        
        # 3. 推理区域（第三列，GPU在上，CPU在下）
        inference_x = section_width * 2.5 - pipeline_width / 2
        # GPU推理管线（上方）
        gpu_y = center_y - pipeline_height / 2 - 5  # 上方，留5像素间距
        self._draw_pipeline(
            painter,
            inference_x,
            gpu_y,
            pipeline_width,
            pipeline_height,
            self.pipeline_data['inference_gpu'],
            job_height,
            job_spacing,
            "GPU推理"
        )
        
        # CPU推理管线（下方）
        cpu_y = center_y + pipeline_height / 2 + 5  # 下方，留5像素间距
        self._draw_pipeline(
            painter,
            inference_x,
            cpu_y,
            pipeline_width,
            pipeline_height,
            self.pipeline_data['inference_cpu'],
            job_height,
            job_spacing,
            "CPU推理"
        )
        
        # 绘制连接线和脉冲效果
        # 转换 -> 队列
        line1_start_x = conversion_x + pipeline_width
        line1_end_x = queue_x
        self._draw_connection_line(
            painter,
            line1_start_x,
            center_y,
            line1_end_x,
            center_y,
            self.pulse_animations['conversion_to_queue']
        )
        
        # 队列 -> GPU推理（从队列上方连接到GPU）
        line2_start_x = queue_x + pipeline_width
        line2_end_x = inference_x
        line2_start_y = center_y - pipeline_height / 4  # 从队列上方连接
        line2_end_y = gpu_y
        self._draw_connection_line(
            painter,
            line2_start_x,
            line2_start_y,
            line2_end_x,
            line2_end_y,
            self.pulse_animations['queue_to_gpu']
        )
        
        # 队列 -> CPU推理（从队列下方连接到CPU）
        line3_start_x = queue_x + pipeline_width
        line3_end_x = inference_x
        line3_start_y = center_y + pipeline_height / 4  # 从队列下方连接
        line3_end_y = cpu_y
        self._draw_connection_line(
            painter,
            line3_start_x,
            line3_start_y,
            line3_end_x,
            line3_end_y,
            self.pulse_animations['queue_to_cpu']
        )
    
    def _draw_pipeline(self, painter, x, y, width, max_height, data, job_height, job_spacing, label):
        """绘制管线（图像转换或推理）"""
        workers = data.get('workers', 0)
        active_jobs = data.get('active_jobs', [])
        
        # 绘制标题
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(COLORS['text_secondary']))
        painter.drawText(int(x), int(y - max_height / 2 - 15), int(width), 20, Qt.AlignCenter, label)
        
        if workers == 0:
            # 没有工作线程，绘制一个空心的占位矩形
            pen = QPen(QColor(COLORS['text_tertiary']), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect_y = y - job_height / 2
            painter.drawRect(int(x), int(rect_y), int(width), int(job_height))
            return
        
        # 计算每个job的位置，确保不超过max_height
        total_height = workers * job_height + (workers - 1) * job_spacing
        # 如果总高度超过限制，缩小job_height和job_spacing
        if total_height > max_height:
            scale = max_height / total_height
            job_height = job_height * scale
            job_spacing = job_spacing * scale
            total_height = max_height
        
        start_y = y - total_height / 2
        
        # 绘制每个job
        for i in range(workers):
            job_y = start_y + i * (job_height + job_spacing)
            is_active = i < len(active_jobs) and active_jobs[i]
            
            if is_active:
                # 实心矩形（有任务）
                brush = QBrush(QColor(COLORS['accent']))
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)
            else:
                # 空心矩形（无任务）
                pen = QPen(QColor(COLORS['text_tertiary']), 1)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
            
            painter.drawRect(int(x), int(job_y), int(width), int(job_height))
    
    def _draw_queue(self, painter, x, y, width, max_height, data, job_height, label):
        """绘制数据队列"""
        queue_size = data.get('size', 0)
        max_size = data.get('max_size', 100)
        
        # 绘制标题
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(COLORS['text_secondary']))
        painter.drawText(int(x), int(y - max_height / 2 - 15), int(width), 20, Qt.AlignCenter, label)
        
        # 计算显示的矩形数量（根据max_height自动调整）
        max_display = int(max_height / (job_height + 2))  # 2是间距
        display_count = min(queue_size, max_display)
        
        if display_count == 0:
            # 没有数据，绘制一个空心的占位矩形
            pen = QPen(QColor(COLORS['text_tertiary']), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect_y = y - job_height / 2
            painter.drawRect(int(x), int(rect_y), int(width), int(job_height))
            return
        
        # 计算每个矩形的位置，确保不超过max_height
        spacing = 2
        total_height = display_count * job_height + (display_count - 1) * spacing
        # 如果总高度超过限制，缩小job_height和spacing
        if total_height > max_height:
            scale = max_height / total_height
            job_height = job_height * scale
            spacing = spacing * scale
            total_height = max_height
        
        start_y = y - total_height / 2
        
        # 绘制队列中的矩形（实心）
        for i in range(display_count):
            job_y = start_y + i * (job_height + spacing)
            # 根据队列大小调整透明度
            alpha = min(255, 100 + int(155 * (queue_size / max(max_size, 1))))
            color = QColor(COLORS['warning'])
            color.setAlpha(alpha)
            brush = QBrush(color)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(x), int(job_y), int(width), int(job_height))
        
        # 如果队列超过显示数量，绘制省略号
        if queue_size > display_count:
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QColor(COLORS['text_tertiary']))
            painter.drawText(int(x), int(start_y + display_count * (job_height + spacing) + 5), 
                           int(width), 15, Qt.AlignCenter, f"+{queue_size - display_count}")
    
    def _draw_connection_line(self, painter, start_x, start_y, end_x, end_y, pulse_value):
        """绘制连接线和脉冲效果"""
        # 基础连接线
        pen = QPen(QColor(COLORS['text_tertiary']), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
        
        # 脉冲效果（如果有数据流动）
        if pulse_value > 0:
            # 计算脉冲位置
            pulse_x = start_x + (end_x - start_x) * pulse_value
            pulse_y = start_y + (end_y - start_y) * pulse_value
            
            # 绘制脉冲圆点
            pulse_radius = 4
            pulse_alpha = int(255 * (1 - abs(pulse_value - 0.5) * 2))  # 中间最亮
            pulse_color = QColor(COLORS['accent'])
            pulse_color.setAlpha(pulse_alpha)
            
            brush = QBrush(pulse_color)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(pulse_x - pulse_radius), int(pulse_y - pulse_radius), 
                              int(pulse_radius * 2), int(pulse_radius * 2))
            
            # 绘制脉冲光晕
            glow_radius = pulse_radius * 2
            glow_alpha = int(100 * (1 - abs(pulse_value - 0.5) * 2))
            glow_color = QColor(COLORS['accent'])
            glow_color.setAlpha(glow_alpha)
            brush = QBrush(glow_color)
            painter.setBrush(brush)
            painter.drawEllipse(int(pulse_x - glow_radius), int(pulse_y - glow_radius),
                              int(glow_radius * 2), int(glow_radius * 2))
