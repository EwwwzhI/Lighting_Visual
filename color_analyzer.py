import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from colour.plotting import plot_chromaticity_diagram_CIE1931, plot_chromaticity_diagram_CIE1976UCS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color
import datetime
import sys
import os

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 优先使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# RGB到XYZ转换矩阵 (更精确的sRGB到XYZ转换矩阵)
RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])

def rgb_to_xyz(rgb):
    """将RGB转换为XYZ颜色空间"""
    # 处理RGBA格式（4个值）或RGB格式（3个值）
    if len(rgb) >= 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        raise ValueError(f"RGB值格式错误: 期望3或4个值，得到{len(rgb)}个值")

    # 线性化sRGB值 (应用gamma校正)
    r, g, b = [c/255.0 for c in (r, g, b)]

    # 应用sRGB gamma校正曲线
    def linearize(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    r_linear = linearize(r)
    g_linear = linearize(g)
    b_linear = linearize(b)

    return np.dot(RGB_TO_XYZ, [r_linear, g_linear, b_linear])

def rgb_to_cie1931(rgb):
    """计算RGB转CIE1931 xy坐标"""
    X, Y, Z = rgb_to_xyz(rgb)
    total = X + Y + Z
    if total == 0:
        return (0.0, 0.0)
    return (round(float(X/total), 2), round(float(Y/total), 2))

def rgb_to_uv(rgb):
    """计算RGB转CIE1976 UCS u'v'坐标"""
    X, Y, Z = rgb_to_xyz(rgb)
    denominator = X + 15*Y + 3*Z
    if denominator == 0:
        return (0.0, 0.0)
    u_prime = 4*X / denominator
    v_prime = 9*Y / denominator
    return (round(float(u_prime), 4), round(float(v_prime), 4))

def rgb_to_lab(rgb):
    """计算RGB转CIELAB坐标"""
    X, Y, Z = rgb_to_xyz(rgb)

    # XYZ值需要乘以100，因为rgb_to_xyz返回的是0-1范围
    X, Y, Z = X * 100, Y * 100, Z * 100

    # 标准光源D65的白色点坐标 (更精确的值)
    Xn, Yn, Zn = 95.047, 100.0, 108.883

    # 归一化
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    # f(t)函数 - 使用更精确的阈值
    def f(t):
        delta = 6.0 / 29.0
        if t > delta**3:
            return t ** (1/3)
        else:
            return (t / (3 * delta**2)) + (4.0 / 29.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    # 计算Lab值
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return (round(float(L), 2), round(float(a), 2), round(float(b), 2))


# 全局变量来跟踪当前打开的图形窗口
current_figure = None
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

def plot_lab_3d_colorspace(points_data, save_path=None):
    """绘制 LAB 三维色度图，并标记选中的点"""
    # 创建图形 - 使用浅灰色背景使白色轴更清晰
    fig = plt.figure(figsize=(18, 16), facecolor='#f0f2f5')
    ax = fig.add_subplot(111, projection='3d', facecolor='#f0f2f5')

    # 设置参数
    max_radius = 100  # 球体半径

    # 创建完整球面的网格
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 80)
    u, v = np.meshgrid(u, v)

    # 将球面坐标转换为笛卡尔坐标
    x = max_radius * np.sin(v) * np.cos(u)  # a轴
    y = max_radius * np.sin(v) * np.sin(u)  # b轴
    z = max_radius * np.cos(v)               # L轴

    # 将xyz坐标映射到LAB颜色空间
    a_values = x
    b_values = y
    L_values = (z + max_radius) / 2

    # 创建颜色数组
    colors_array = np.zeros((*a_values.shape, 4))

    # 计算每个点的颜色
    for i in range(a_values.shape[0]):
        for j in range(a_values.shape[1]):
            L = L_values[i, j]
            a = a_values[i, j]
            b = b_values[i, j]

            angle_from_top = v[i, j]
            alpha = 0.75 - 0.2 * np.sin(angle_from_top)

            try:
                lab = np.array([[[L, a, b]]])
                rgb = color.lab2rgb(lab)
                colors_array[i, j, :3] = rgb[0, 0]
                colors_array[i, j, 3] = alpha
            except:
                colors_array[i, j] = [0.9, 0.9, 0.9, alpha]

    # 绘制主球体
    surf = ax.plot_surface(x, y, z, facecolors=colors_array,
                           shade=True, antialiased=True,
                           linewidth=0, rcount=120, ccount=80)

    # 绘制更精细的参考圆
    circle_points = 150
    theta = np.linspace(0, 2*np.pi, circle_points)

    # XY平面的圆（a-b平面）
    for r in [max_radius * 0.5, max_radius]:
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        z_circle = np.zeros_like(theta)
        ax.plot(x_circle, y_circle, z_circle, color='#888888', linewidth=1.2, alpha=0.4, linestyle='--')

    # 绘制更粗的坐标轴
    axis_length = max_radius * 1.5

    # a轴 (绿到红)
    ax.plot([-axis_length, 0], [0, 0], [0, 0], color='#2ecc71', linewidth=4, alpha=0.95, linestyle='-')
    ax.plot([0, axis_length], [0, 0], [0, 0], color='#e74c3c', linewidth=4, alpha=0.95, linestyle='-')

    # b轴 (蓝到黄)
    ax.plot([0, 0], [-axis_length, 0], [0, 0], color='#3498db', linewidth=4, alpha=0.95, linestyle='-')
    ax.plot([0, 0], [0, axis_length], [0, 0], color='#f39c12', linewidth=4, alpha=0.95, linestyle='-')

    # L轴 (黑到白)
    ax.plot([0, 0], [0, 0], [-axis_length, 0], color='#2c3e50', linewidth=4, alpha=0.95, linestyle='-')
    ax.plot([0, 0], [0, 0], [0, axis_length], color='#ecf0f1', linewidth=4, alpha=0.95, linestyle='-')

    # 添加更美观的标签
    label_distance = axis_length * 1.15

    # 使用更大、更美观的标签框
    ax.text(label_distance, 0, 0, 'Red\n+a', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e74c3c', alpha=0.85, edgecolor='white', linewidth=2),
            color='white', weight='bold')
    ax.text(-label_distance, 0, 0, 'Green\n-a', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#2ecc71', alpha=0.85, edgecolor='white', linewidth=2),
            color='white', weight='bold')
    ax.text(0, label_distance, 0, 'Yellow\n+b', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f39c12', alpha=0.9, edgecolor='white', linewidth=2),
            color='white', weight='bold')
    ax.text(0, -label_distance, 0, 'Blue\n-b', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#3498db', alpha=0.85, edgecolor='white', linewidth=2),
            color='white', weight='bold')
    ax.text(0, 0, label_distance, 'White\nL=100', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='#2c3e50', linewidth=3),
            color='#2c3e50', weight='bold')
    ax.text(0, 0, -label_distance, 'Black\nL=0', fontsize=16, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#2c3e50', alpha=0.9, edgecolor='white', linewidth=2),
            color='white', weight='bold')

    # 标记选中的点 - 更清晰的显示
    if points_data:
        for i, point in enumerate(points_data):
            L, a, b_val = point['lab']
            # 将 LAB 转换为 3D 坐标
            z_point = L * 2 - 100
            a_point = a
            b_point = b_val

            # 获取点的RGB颜色
            rgb = point['rgb']
            point_color = '#{:02x}{:02x}{:02x}'.format(*rgb)

            # 绘制显眼的点标记 - 黑色外圈
            ax.scatter([a_point], [b_point], [z_point],
                      color='none', s=400, marker='o',
                      edgecolors='black', linewidths=3, zorder=99, alpha=1.0)

            # 白色中圈
            ax.scatter([a_point], [b_point], [z_point],
                      color='none', s=320, marker='o',
                      edgecolors='white', linewidths=3, zorder=100, alpha=1.0)

            # 实际颜色点作为背景
            ax.scatter([a_point], [b_point], [z_point],
                      color=point_color, s=250, marker='o',
                      edgecolors='none', linewidths=0, zorder=101, alpha=1.0)

            # 计算反色用于字体显示
            contrast_color = get_contrast_color(rgb)

            # 在点的最中间显示序号 - 使用反色字体
            ax.text(a_point, b_point, z_point, f'{i+1}',
                   fontsize=9, fontweight='bold', color=contrast_color,
                   zorder=102, ha='center', va='center')

    # 设置坐标轴标签 - 中文标签
    ax.set_xlabel('a 轴 (绿 ← → 红)', fontsize=16, weight='bold', labelpad=20, color='#34495e')
    ax.set_ylabel('b 轴 (蓝 ← → 黄)', fontsize=16, weight='bold', labelpad=20, color='#34495e')
    ax.set_zlabel('L 轴 (亮度)', fontsize=16, weight='bold', labelpad=20, color='#34495e')

    # 固定视角，不可交互
    ax.view_init(elev=25, azim=45)

    lim = axis_length * 0.92
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # 优化网格和背景
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    ax.xaxis.pane.set_edgecolor('#bdc3c7')
    ax.yaxis.pane.set_edgecolor('#bdc3c7')
    ax.zaxis.pane.set_edgecolor('#bdc3c7')

    # 设置刻度
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_zticks([-100, -50, 0, 50, 100])

    # 刻度标签样式
    ax.tick_params(labelsize=11, colors='#34495e')

    # 更美观的标题
    title = 'CIE LAB 色彩空间 - 三维可视化'
    if points_data:
        title += f'\n已选择 {len(points_data)} 个点'
    ax.set_title(title, fontsize=22, weight='bold', pad=35, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6', linewidth=2))

    # 添加说明文字
    fig.text(0.5, 0.02,
             'LAB 色彩空间：L (亮度 0-100) | a (绿- 到 红+) | b (蓝- 到 黄+)',
             ha='center', fontsize=12, style='italic', color='#7f8c8d',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7', linewidth=1.5))

    # 调整布局，增加左右边距避免截断
    plt.tight_layout(rect=[0.05, 0.04, 0.95, 0.98])

    if save_path:
        # 保存时增加边距，避免坐标轴标签被截断
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   pad_inches=0.5, facecolor='#f0f2f5')
        print(f"LAB 3D色度图已保存到: {save_path}")

    plt.show()
    return fig


def plot_combined_chromaticity_diagrams(points_data, save_path=None):
    """绘制合并的色度图（CIE1931和CIE1976 UCS）"""
    # 检查数据是否为空
    if not points_data:
        messagebox.showwarning("无数据", "请先选择至少一个点")
        return None

    global current_figure

    if current_figure is not None:
        plt.close(current_figure)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor='white')
    current_figure = fig

    # 设置坐标轴背景为白色
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # CIE 1931色度图
    plot_chromaticity_diagram_CIE1931(axes=ax1, show=False)
    for i, point in enumerate(points_data):
        xy = point['cie1931']
        # 使用互补色作为标记颜色
        rgb = point['rgb']
        marker_color = get_contrast_color(rgb)

        # 绘制标记点：实心圆+白色边框，专业且清晰
        ax1.plot(xy[0], xy[1],
                marker='o',
                color=marker_color,
                markersize=7,
                markeredgewidth=1.5,
                markeredgecolor='white',  # 白色边框提高可见度
                markerfacecolor=marker_color,  # 实心填充
                zorder=10)  # 确保标记在最上层

        # 添加数字标签 - 智能定位避免遮挡
        # 根据点的位置决定标签偏移方向
        offset_x = 0.02 if xy[0] < 0.5 else -0.02
        offset_y = 0.02 if xy[1] < 0.5 else -0.02
        ha = 'left' if xy[0] < 0.5 else 'right'
        va = 'bottom' if xy[1] < 0.5 else 'top'

        ax1.text(xy[0] + offset_x, xy[1] + offset_y, str(i+1),
                fontsize=7, fontweight='bold',
                color=marker_color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                         edgecolor=marker_color, linewidth=1, alpha=0.9),
                zorder=11, ha=ha, va=va)

    ax1.set_title('CIE 1931 色度图', fontsize=14, fontweight='bold')
    # 创建自定义图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=get_contrast_color(point['rgb']),
                                  markeredgecolor='white', markeredgewidth=1.5,
                                  markersize=7, label=f'点 {i+1}')
                      for i, point in enumerate(points_data)]
    ax1.legend(handles=legend_elements, fontsize=9, loc='best')

    # CIE 1976 UCS色度图
    plot_chromaticity_diagram_CIE1976UCS(axes=ax2, show=False)
    for i, point in enumerate(points_data):
        uv = point['uv']
        # 使用互补色作为标记颜色
        rgb = point['rgb']
        marker_color = get_contrast_color(rgb)

        # 绘制标记点：实心圆+白色边框
        ax2.plot(uv[0], uv[1],
                marker='o',
                color=marker_color,
                markersize=7,
                markeredgewidth=1.5,
                markeredgecolor='white',
                markerfacecolor=marker_color,
                zorder=10)

        # 添加数字标签 - 智能定位避免遮挡
        # 根据点的位置决定标签偏移方向
        offset_x = 0.015 if uv[0] < 0.5 else -0.015
        offset_y = 0.015 if uv[1] < 0.5 else -0.015
        ha = 'left' if uv[0] < 0.5 else 'right'
        va = 'bottom' if uv[1] < 0.5 else 'top'

        ax2.text(uv[0] + offset_x, uv[1] + offset_y, str(i+1),
                fontsize=7, fontweight='bold',
                color=marker_color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                         edgecolor=marker_color, linewidth=1, alpha=0.9),
                zorder=11, ha=ha, va=va)

    ax2.set_title('CIE 1976 UCS 色度图', fontsize=14, fontweight='bold')
    # 创建自定义图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=get_contrast_color(point['rgb']),
                                  markeredgecolor='white', markeredgewidth=1.5,
                                  markersize=7, label=f'点 {i+1}')
                      for i, point in enumerate(points_data)]
    ax2.legend(handles=legend_elements, fontsize=9, loc='best')

    # 先使用 tight_layout，然后强制设置子图间距
    plt.tight_layout(pad=1.5)
    # 在tight_layout之后再次设置间距，紧凑的间隔
    fig.subplots_adjust(wspace=0.1, left=0.05, right=0.95)

    # 强制设置整个figure的背景为白色（colour库可能会修改）
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    if save_path:
        # 保存时确保整个图片包括坐标轴和标题都有白色背景
        # 不使用 bbox_inches='tight' 以保留完整背景
        fig.savefig(save_path, dpi=300, facecolor='white', edgecolor='none')
        print(f"合并色度图已保存到: {save_path}")

    plt.show()
    return current_figure

def save_file_dialog(filetype, default_name, title):
    """通用文件保存对话框"""
    return filedialog.asksaveasfilename(
        title=title,
        defaultextension=".png",
        filetypes=filetype,
        initialfile=default_name
    )

def generate_filename(prefix):
    """自动生成带时间戳的文件名"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


class ImageViewer(ttk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        colors = setup_modern_ui()
        self.bg_color = colors['bg_primary']

        self.canvas = tk.Canvas(self, bg=self.bg_color, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        self.original_image = None
        self.clean_original_image = None
        self.current_image = None
        self.scale_factor = 1.0
        self.initial_fit_done = False
        self.click_callback = None

    def set_image(self, image, reset_zoom=True, save_clean_copy=True):
        if image is None:
            raise ValueError("图像对象为空")

        if not hasattr(image, 'size'):
            raise ValueError("提供的对象不是有效的图像")

        self.original_image = image
        if save_clean_copy:
            self.clean_original_image = image.copy()

        if reset_zoom:
            self.current_image = image.copy()
            self.scale_factor = 1.0
            self.initial_fit_done = False
        else:
            self.resize_image()

        self.display_image()
        if not self.initial_fit_done:
            self.after(100, self.auto_fit_image)
            self.initial_fit_done = True

    def display_image(self):
        if self.current_image:
            self.tk_image = ImageTk.PhotoImage(self.current_image)
            self.canvas.delete("all")
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_click(self, event):
        if not self.original_image or not self.click_callback:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        original_x = int(canvas_x / self.scale_factor)
        original_y = int(canvas_y / self.scale_factor)

        if (0 <= original_x < self.original_image.width and
            0 <= original_y < self.original_image.height):
            self.click_callback(original_x, original_y)

    def auto_fit_image(self):
        if not self.original_image:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.after(100, self.auto_fit_image)
            return

        img_width, img_height = self.original_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)

        self.resize_image()

    def resize_image(self):
        if self.original_image:
            img_width, img_height = self.original_image.size
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            self.current_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.display_image()

    def zoom_in(self):
        if self.original_image and self.scale_factor < 3.0:
            self.scale_factor *= 1.2
            self.resize_image()

    def zoom_out(self):
        if self.original_image and self.scale_factor > 0.1:
            self.scale_factor /= 1.2
            self.resize_image()


def setup_modern_ui():
    """创建现代化UI主题"""
    style = ttk.Style()
    style.theme_use('clam')

    colors = {
        'bg_primary': '#2b2b2b',
        'bg_secondary': '#3c3c3c',
        'bg_accent': '#404040',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'accent_blue': '#4a9eff',
        'accent_green': '#4ecdc4',
        'accent_orange': '#ff9f43',
        'accent_red': '#ff6b6b',
        'border': '#555555',
    }

    # 基础样式
    style.configure('TFrame', background=colors['bg_primary'])
    style.configure('TLabelframe', background=colors['bg_secondary'], foreground=colors['text_primary'])
    style.configure('TLabelframe.Label', background=colors['bg_secondary'], foreground=colors['text_primary'], font=('Segoe UI', 10, 'bold'))

    # 按钮样式
    button_styles = {
        'Modern': colors['bg_accent'],
        'Primary': colors['accent_blue'],
        'Success': colors['accent_green'],
        'Danger': colors['accent_red'],
        'Warning': colors['accent_orange'],
        'Info': colors['accent_blue'],
    }

    for style_name, color in button_styles.items():
        is_bold = style_name in ['Primary', 'Success', 'Danger', 'Warning', 'Info']
        font = ('Segoe UI', 9, 'bold') if is_bold else ('Segoe UI', 9)
        style.configure(f'{style_name}.TButton',
                      background=color,
                      foreground=colors['text_primary'],
                      borderwidth=0,
                      focuscolor='none',
                      padding=(12, 8),
                      font=font)

    # 标签样式
    style.configure('Title.TLabel', background=colors['bg_primary'], foreground=colors['text_primary'], font=('Segoe UI', 12, 'bold'))
    style.configure('Modern.TLabel', background=colors['bg_secondary'], foreground=colors['text_primary'], font=('Segoe UI', 9))
    style.configure('Secondary.TLabel', background=colors['bg_secondary'], foreground=colors['text_secondary'], font=('Segoe UI', 8))

    # Treeview样式
    style.configure('Modern.Treeview',
                   background=colors['bg_primary'],
                   foreground=colors['text_primary'],
                   fieldbackground=colors['bg_primary'],
                   borderwidth=0,
                   rowheight=30,  # 增加行高，使行间距更大
                   font=('Segoe UI', 11))

    style.configure('Modern.Treeview.Heading',
                   background=colors['accent_blue'],
                   foreground=colors['text_primary'],
                   font=('Segoe UI', 12, 'bold'),
                   relief='flat')

    style.map('Modern.Treeview',
              background=[('selected', colors['accent_blue'])],
              foreground=[('selected', 'white')])

    return colors




def create_toolbar_modern(parent, root, points_data, image_viewer, info_panel, color_palette_updater):
    """创建现代化工具栏 - 增大按钮尺寸"""
    colors = setup_modern_ui()

    # 大尺寸工具栏设计
    toolbar = tk.Frame(parent, bg=colors['bg_primary'], relief='ridge', bd=2, height=120)
    toolbar.pack(fill=tk.X, padx=10, pady=8)
    toolbar.pack_propagate(False)

    # 工具栏内部容器
    inner_frame = tk.Frame(toolbar, bg=colors['bg_secondary'])
    inner_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # 内容区域
    content = tk.Frame(inner_frame, bg=colors['bg_secondary'])
    content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

    # 按钮组定义 - 统一海洋蓝配色方案
    button_groups = [
        ("分析", [
            ("清除所有点", '#2874a6', lambda: clear_points(image_viewer, points_data, info_panel, color_palette_updater)),  # 海洋蓝
            ("显示点信息", '#2874a6', lambda: show_points_info(points_data)),  # 海洋蓝
            ("CIE色度坐标图", '#2874a6', lambda: show_combined_plot(points_data)),  # 海洋蓝
            ("LAB色彩空间图", '#2874a6', lambda: show_lab_3d_plot(points_data))  # 海洋蓝
        ]),
        ("保存", [
            ("保存标记图像", '#2874a6', lambda: save_image_with_markers(image_viewer, points_data)),  # 海洋蓝
            ("保存CIE坐标图", '#2874a6', lambda: save_chromaticity_plot(points_data)),  # 海洋蓝
            ("保存LAB空间图", '#2874a6', lambda: save_lab_3d_plot(points_data))  # 海洋蓝
        ]),
        ("图片", [
            ("更改图片", '#2874a6', lambda: change_image(root, points_data, image_viewer, info_panel, color_palette_updater))  # 海洋蓝
        ]),
        ("视图", [
            ("放大 +", '#2874a6', lambda: image_viewer.zoom_in()),  # 海洋蓝
            ("缩小 -", '#2874a6', lambda: image_viewer.zoom_out())  # 海洋蓝
        ])
    ]

    # 创建专业按钮组
    for i, (group_name, buttons) in enumerate(button_groups):
        group = tk.Frame(content, bg=colors['bg_secondary'])
        group.pack(side=tk.LEFT, padx=(0, 30 if i < len(button_groups) - 1 else 0))

        # 专业组标题设计
        title_frame = tk.Frame(group, bg=colors['bg_secondary'])
        title_frame.pack(fill=tk.X, pady=(0, 10))

        # 添加标题装饰线
        deco_line = tk.Frame(title_frame, bg=colors['accent_blue'], height=2)
        deco_line.pack(fill=tk.X, pady=(0, 4))

        tk.Label(title_frame, text=group_name, bg=colors['bg_secondary'],
                fg=colors['accent_blue'], font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)

        btn_container = tk.Frame(group, bg=colors['bg_secondary'])
        btn_container.pack()

        for i, (text, color, command) in enumerate(buttons):
            is_bold = group_name in ["分析", "保存"]
            # 统一海洋蓝配色方案的大尺寸按钮设计
            # 海洋蓝的悬停颜色（更亮的版本）
            active_bg = '#3498db'  # 海洋蓝悬停

            btn = tk.Button(btn_container, text=text, command=command,
                          bg=color, fg=colors['text_primary'],
                          font=('Segoe UI', 12, 'bold') if is_bold else ('Segoe UI', 12),
                          relief='flat', bd=0, padx=35, pady=18, cursor='hand2',
                          activebackground=active_bg, activeforeground='white',
                          highlightbackground=color, highlightthickness=0)
            btn.pack(side=tk.LEFT, padx=(0 if i == 0 else 15, 0))

            # 统一的鼠标悬停效果
            def on_enter(e, btn=btn):
                btn.config(bg='#3498db')  # 海洋蓝悬停

            def on_leave(e, btn=btn, original_bg=color):
                btn.config(bg=original_bg)

            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

    # 在工具栏右下角添加快捷键提示
    shortcut_hint = tk.Label(content,
                             text="快捷键：Ctrl + 滚轮 缩放   方向键 移动图片",
                             bg=colors['bg_secondary'],
                             fg=colors['text_secondary'],
                             font=('Segoe UI', 10),
                             padx=15, pady=5)
    shortcut_hint.pack(side=tk.RIGHT, anchor='se')

    return toolbar


def create_info_panel_modern(parent, points_data, colors):
    """创建平衡的信息面板 - 5:5比例中的5部分"""
    main_frame = tk.Frame(parent, bg=colors['bg_secondary'])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

    # 标题 - 居中显示
    tk.Label(main_frame, text="数据分析", bg=colors['bg_secondary'],
            fg=colors['text_primary'], font=('Segoe UI', 16, 'bold')).pack(pady=(0, 8))

    # 点数统计 - 适中的布局
    stats_container = tk.Frame(main_frame, bg=colors['bg_accent'])
    stats_container.pack(fill=tk.X, pady=(0, 8))

    count_container = tk.Frame(stats_container, bg=colors['bg_accent'])
    count_container.pack(fill=tk.X, padx=15, pady=10)

    # 点数显示 - 大幅增加高度和字体
    count_frame = tk.Frame(count_container, bg=colors['accent_blue'], width=170, height=90)
    count_frame.pack(side=tk.LEFT, padx=(0, 20))
    count_frame.pack_propagate(False)

    count_label = tk.Label(count_frame, text="0", bg=colors['accent_blue'],
                          fg=colors['text_primary'], font=('Segoe UI', 46, 'bold'))
    count_label.pack(expand=True)

    # "/10"标签与数字对齐
    slash_label = tk.Label(count_container, text="/10", bg=colors['bg_accent'],
                          fg=colors['text_secondary'], font=('Segoe UI', 24, 'bold'))
    slash_label.pack(side=tk.LEFT, padx=(0, 22))

    # 状态信息
    status_label = tk.Label(count_container, text="就绪", bg=colors['bg_accent'],
                           fg=colors['text_secondary'], font=('Segoe UI', 12))
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # 分析文本区域 - 固定高度不扩展
    tk.Label(main_frame, text="点分析", bg=colors['bg_secondary'],
            fg=colors['text_primary'], font=('Segoe UI', 14, 'bold')).pack(pady=(4, 4))

    text_frame = tk.Frame(main_frame, bg=colors['bg_accent'])
    text_frame.pack(fill=tk.X, expand=False)

    # 创建Treeview表格
    tree_frame = tk.Frame(text_frame, bg=colors['bg_accent'])
    tree_frame.pack(fill=tk.X, expand=False)

    # 定义列
    columns = ("point", "position", "rgb", "cie1931", "cie1976", "lab")

    # 创建Treeview - 显示10个点，调整为10行刚好显示所有点
    treeview = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10, style="Modern.Treeview")

    # 设置列标题和宽度
    treeview.heading("point", text="点")
    treeview.heading("position", text="位置")
    treeview.heading("rgb", text="RGB")
    treeview.heading("cie1931", text="CIE xy")
    treeview.heading("cie1976", text="CIE u'v'")
    treeview.heading("lab", text="CIELAB")

    # 设置列宽 - 重新分配宽度保持总宽度不超
    treeview.column("point", width=30, anchor="center")
    treeview.column("position", width=100, anchor="center")
    treeview.column("rgb", width=100, anchor="center")
    treeview.column("cie1931", width=90, anchor="center")
    treeview.column("cie1976", width=90, anchor="center")
    treeview.column("lab", width=155, anchor="center")

    # 确保表格有固定高度和显示
    treeview.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    return {
        'points_label': count_label,
        'recent_points_text': treeview,
        'status_label': status_label
    }


def create_color_palette_modern(parent, points_data, colors):
    """创建简化的色彩调色板"""
    main_frame = tk.Frame(parent, bg=colors['bg_secondary'])
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 10))

    # 添加顶部分隔线
    separator = tk.Frame(main_frame, bg=colors['border'], height=1)
    separator.pack(fill=tk.X, pady=(0, 6))

    tk.Label(main_frame, text="色彩调色板", bg=colors['bg_secondary'],
            fg=colors['text_primary'], font=('Segoe UI', 16, 'bold')).pack(pady=(0, 6))

    # 内容区域 - 填充固定空间，但内容靠顶部对齐
    content_frame = tk.Frame(main_frame, bg=colors['bg_accent'])
    content_frame.pack(fill=tk.BOTH, expand=True)

    def update_palette():
        for widget in content_frame.winfo_children():
            widget.destroy()

        if not points_data:
            # 创建更丰富的提示信息
            tip_frame = tk.Frame(content_frame, bg=colors['bg_accent'])
            tip_frame.pack(pady=50)

            # 提示图标
            tk.Label(tip_frame, text="🎨", bg=colors['bg_accent'], fg=colors['text_secondary'],
                    font=('Segoe UI', 24)).pack()

            # 提示文字
            tk.Label(tip_frame, text="尚未选择任何点", bg=colors['bg_accent'], fg=colors['text_secondary'],
                    font=('Segoe UI', 12, 'bold')).pack(pady=(10, 5))

            tk.Label(tip_frame, text="点击左侧图片选择颜色点", bg=colors['bg_accent'], fg=colors['text_secondary'],
                    font=('Segoe UI', 10)).pack()
            return

        # 创建双列布局容器 - 靠顶部对齐，不扩展高度
        columns_frame = tk.Frame(content_frame, bg=colors['bg_accent'])
        columns_frame.pack(side=tk.TOP, anchor='n', fill=tk.X, expand=False, padx=2, pady=2)

        # 左列和右列 - 靠顶部对齐，防止高度不一致时位移
        left_column = tk.Frame(columns_frame, bg=colors['bg_accent'])
        left_column.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor='n', padx=(0, 1))

        right_column = tk.Frame(columns_frame, bg=colors['bg_accent'])
        right_column.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor='n', padx=(1, 0))

        for i, point in enumerate(points_data):
            color = point['rgb']
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color)

            # 确定列位置
            target_column = left_column if i % 2 == 0 else right_column

    # 创建颜色卡片 - 进一步紧凑布局以便10个点完全显示无需滚动
            card_frame = tk.Frame(target_column, bg=colors['bg_secondary'])
            card_frame.pack(pady=2, anchor=tk.W)

            # 颜色预览和信息 - 紧凑内边距
            preview_frame = tk.Frame(card_frame, bg=colors['bg_secondary'])
            preview_frame.pack(padx=3, pady=2, anchor=tk.W)

            # 颜色方块 - 适中尺寸
            color_canvas = tk.Canvas(preview_frame, width=85, height=85,
                                   bg=color_hex, highlightthickness=2,
                                   highlightbackground=colors['border'])
            color_canvas.pack(side=tk.LEFT, padx=(0, 6))

            # 点信息和坐标数据
            info_frame = tk.Frame(preview_frame, bg=colors['bg_secondary'])
            info_frame.pack(side=tk.LEFT, padx=(5, 0), fill=tk.BOTH)

            # 点编号和RGB信息 - 增大字体
            header_label = tk.Label(info_frame, text=f"#{i+1} RGB({color[0]},{color[1]},{color[2]})",
                                   bg=colors['bg_secondary'], fg=colors['text_primary'],
                                   font=('Segoe UI', 11, 'bold'))
            header_label.pack(anchor=tk.W)

            # 坐标信息 - 删除CIE前缀，显示两位小数
            coord_text = f"xy: {point['cie1931'][0]:.2f}, {point['cie1931'][1]:.2f}"
            coord_label1 = tk.Label(info_frame, text=coord_text,
                                   bg=colors['bg_secondary'], fg=colors['text_secondary'],
                                   font=('Consolas', 10))
            coord_label1.pack(anchor=tk.W)

            coord_text2 = f"u'v': {point['uv'][0]:.3f}, {point['uv'][1]:.3f}"
            coord_label2 = tk.Label(info_frame, text=coord_text2,
                                   bg=colors['bg_secondary'], fg=colors['text_secondary'],
                                   font=('Consolas', 10))
            coord_label2.pack(anchor=tk.W)

            coord_text3 = f"LAB: {point['lab'][0]:.2f}, {point['lab'][1]:.2f}, {point['lab'][2]:.2f}"
            coord_label3 = tk.Label(info_frame, text=coord_text3,
                                   bg=colors['bg_secondary'], fg=colors['text_secondary'],
                                   font=('Consolas', 10))
            coord_label3.pack(anchor=tk.W)

    return update_palette


def on_click(x, y, image_viewer, points_data, info_panel, color_palette_updater):
    """处理鼠标点击事件"""
    # 检查点数限制
    if len(points_data) >= 10:
        messagebox.showwarning("达到最大点数", "已达到最大选择点数（10个），无法添加更多点。")
        return

    # 检查图像是否存在 - 使用clean_original_image确保获取原始颜色
    if image_viewer.clean_original_image is None:
        messagebox.showwarning("无图像", "请先加载一张图像")
        return

    # 检查坐标范围
    if x < 0 or y < 0:
        messagebox.showwarning("坐标错误", "坐标不能为负数")
        return

    width, height = image_viewer.clean_original_image.size
    if x >= width or y >= height:
        messagebox.showwarning("坐标错误", f"坐标超出图像范围 (0-{width-1}, 0-{height-1})")
        return

    # 从干净的原图获取像素值，避免获取到标记的颜色
    rgb = image_viewer.clean_original_image.getpixel((x, y))

    # 确保RGB值是元组格式（处理可能的整数返回值）
    if not isinstance(rgb, tuple):
        rgb = (rgb, rgb, rgb)  # 灰度图像转换为RGB

    # 只取前3个值（忽略alpha通道）
    if len(rgb) >= 3:
        rgb = rgb[:3]
    else:
        messagebox.showerror("数据错误", f"无法获取有效的RGB值，得到{len(rgb)}个值")
        return

    # 计算颜色坐标
    cie1931 = rgb_to_cie1931(rgb)
    uv = rgb_to_uv(rgb)
    lab = rgb_to_lab(rgb)

    point_info = {
        'x': x, 'y': y, 'rgb': rgb,
        'cie1931': cie1931,
        'uv': uv,
        'lab': lab
    }
    points_data.append(point_info)

    # 更新界面
    update_image_display(image_viewer, points_data)
    info_panel['points_label'].config(text=str(len(points_data)))
    update_recent_points_info(points_data, info_panel['recent_points_text'])
    info_panel['status_label'].config(text=f"已添加点 {len(points_data)}")
    color_palette_updater()

    print(f"点 {len(points_data)}: 位置({x},{y}) RGB{rgb} CIE_xy{cie1931} CIE_uv{uv} CIELAB{lab}")


def update_recent_points_info(points_data, treeview):
    """更新点分析信息 - 使用Treeview表格显示"""
    # 清除现有数据
    for item in treeview.get_children():
        treeview.delete(item)

    if not points_data:
        # 重新分配宽度保持总宽度不超
        treeview.column("point", width=30, anchor="center")
        treeview.column("position", width=100, anchor="center")
        treeview.column("rgb", width=100, anchor="center")
        treeview.column("cie1931", width=90, anchor="center")
        treeview.column("cie1976", width=90, anchor="center")
        treeview.column("lab", width=155, anchor="center")
        # 添加一个空行确保表格结构可见
        treeview.insert("", "end", values=("", "", "", "", "", ""))
        return

    # 清除可能的空行
    for item in treeview.get_children():
        treeview.delete(item)

    # 插入数据行
    for i, point in enumerate(points_data):
        # 优化格式化数据 - 添加括号使其更清晰
        point_num = f"{i + 1}"
        position = f"({point['x']}, {point['y']})"
        rgb = f"({point['rgb'][0]}, {point['rgb'][1]}, {point['rgb'][2]})"
        cie1931 = f"({point['cie1931'][0]:.2f}, {point['cie1931'][1]:.2f})"
        cie1976 = f"({point['uv'][0]:.2f}, {point['uv'][1]:.2f})"
        lab = f"({point['lab'][0]:.1f}, {point['lab'][1]:.1f}, {point['lab'][2]:.1f})"

        # 调试信息 - 确保lab数据存在
        print(f"Point {i+1}: lab={lab}")

        # 插入行数据
        treeview.insert("", "end", values=(point_num, position, rgb, cie1931, cie1976, lab))


def get_contrast_color(rgb):
    """计算RGB的互补色（反转颜色）作为标记颜色"""
    # 简单的颜色反转：255 - 原值
    r_inv = 255 - rgb[0]
    g_inv = 255 - rgb[1]
    b_inv = 255 - rgb[2]

    return '#{:02x}{:02x}{:02x}'.format(r_inv, g_inv, b_inv)


def update_image_display(image_viewer, points_data):
    """更新图像显示，绘制标记点"""
    if not points_data:
        if image_viewer.clean_original_image:
            image_viewer.set_image(image_viewer.clean_original_image, reset_zoom=False, save_clean_copy=False)
        return

    img_work_copy = image_viewer.clean_original_image.copy() if image_viewer.clean_original_image else image_viewer.original_image.copy()
    draw = ImageDraw.Draw(img_work_copy)

    # 固定标记大小，确保在不同图片上视觉大小一致
    radius = 10  # 固定半径10像素
    line_width = 3  # 固定线宽3像素

    # 尝试加载更大的字体，使文字更显眼
    try:
        # 尝试使用系统字体，字号16
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            # 如果arial不可用，尝试其他常见字体
            font = ImageFont.truetype("segoeui.ttf", 16)
        except:
            # 如果都不可用，使用默认字体
            font = ImageFont.load_default()

    for i, point in enumerate(points_data):
        # 根据点的颜色动态生成对比色
        marker_color = get_contrast_color(point['rgb'])

        # 绘制标记圆圈
        draw.ellipse([point['x'] - radius, point['y'] - radius,
                     point['x'] + radius, point['y'] + radius],
                    outline=marker_color, width=line_width)

        # 绘制点编号 - 居中显示
        text = str(i+1)
        # 获取文字边界框以计算居中位置
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 计算居中位置
        text_x = point['x'] - text_width // 2
        text_y = point['y'] - text_height // 2

        draw.text((text_x, text_y), text, fill=marker_color, font=font)

    image_viewer.set_image(img_work_copy, reset_zoom=False, save_clean_copy=False)
    return img_work_copy


def clear_points(image_viewer, points_data, info_panel, color_palette_updater):
    """清除所有点"""
    points_data.clear()

    if image_viewer.clean_original_image:
        image_viewer.set_image(image_viewer.clean_original_image, reset_zoom=False, save_clean_copy=False)

    info_panel['points_label'].config(text="0")
    update_recent_points_info(points_data, info_panel['recent_points_text'])
    info_panel['status_label'].config(text="已清除所有点")
    color_palette_updater()

    messagebox.showinfo("清除", "已清除所有标记点")


def change_image(root, points_data, image_viewer, info_panel, color_palette_updater):
    """更改图片功能"""
    filepath = filedialog.askopenfilename(
        title="选择新图片",
        filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if not filepath:
        return

    # 验证文件是否存在
    if not os.path.exists(filepath):
        messagebox.showerror("文件错误", "选择的文件不存在")
        return

    # 清除现有点
    points_data.clear()

    # 尝试打开图像
    img = Image.open(filepath)

    # 验证图像格式
    if img.format not in ['PNG', 'JPEG', 'BMP', 'GIF']:
        # PIL可能会自动转换格式，这里只检查基本的图像属性
        pass

    # 设置图像
    image_viewer.set_image(img, reset_zoom=True, save_clean_copy=True)

    # 更新界面
    info_panel['points_label'].config(text="0")
    update_recent_points_info(points_data, info_panel['recent_points_text'])
    info_panel['status_label'].config(text="已加载新图片")
    color_palette_updater()

    # 调整窗口大小
    img_width, img_height = img.size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width, window_height, x, y = get_window_size(img_width, img_height, screen_width, screen_height)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    messagebox.showinfo("更改图片", "已成功加载新图片")

def save_image_with_markers(image_viewer, points_data):
    """保存带标记的图像"""
    if not points_data:
        messagebox.showwarning("无数据", "没有可保存的标记点")
        return

    filepath = save_file_dialog([("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")],
                               generate_filename("marked_image"), "保存标记图像")
    if filepath:
        img_marked = update_image_display(image_viewer, points_data)
        img_marked.save(filepath)
        messagebox.showinfo("保存成功", f"标记图像已保存到:\n{filepath}")

def save_chromaticity_plot(points_data):
    """保存色度图"""
    if not points_data:
        messagebox.showwarning("无数据", "没有可保存的点数据")
        return

    filepath = save_file_dialog([("PNG文件", "*.png"), ("PDF文件", "*.pdf"), ("所有文件", "*.*")],
                               generate_filename("combined_chromaticity"), "保存色度图")
    if filepath:
        plot_combined_chromaticity_diagrams(points_data, save_path=filepath)
        messagebox.showinfo("保存成功", f"色度图已保存到:\n{filepath}")

def show_points_info(points_data):
    """显示所有点信息 - 使用三列卡片式布局"""
    if not points_data:
        messagebox.showinfo("点信息", "尚未选择任何点")
        return

    # 创建自定义窗口
    info_window = tk.Toplevel()
    info_window.title(f"分析点信息 - 总共 {len(points_data)} 个点")
    info_window.geometry("850x850")

    colors = setup_modern_ui()
    info_window.configure(bg=colors['bg_primary'])

    # 标题栏
    title_frame = tk.Frame(info_window, bg=colors['bg_secondary'], height=70)
    title_frame.pack(fill=tk.X, padx=10, pady=10)
    title_frame.pack_propagate(False)

    tk.Label(title_frame, text=f"📊 分析点详细信息",
             bg=colors['bg_secondary'], fg=colors['text_primary'],
             font=('Segoe UI', 18, 'bold')).pack(side=tk.LEFT, padx=20, pady=20)

    tk.Label(title_frame, text=f"共 {len(points_data)} 个点",
             bg=colors['bg_secondary'], fg=colors['text_secondary'],
             font=('Segoe UI', 12)).pack(side=tk.LEFT, padx=(0, 20), pady=20)

    # 创建可滚动的画布容器
    canvas_frame = tk.Frame(info_window, bg=colors['bg_primary'])
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    canvas = tk.Canvas(canvas_frame, bg=colors['bg_primary'], highlightthickness=0)
    scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=colors['bg_primary'])

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
    canvas.configure(yscrollcommand=scrollbar.set)

    # 创建三列容器 - 保持高度对齐
    left_column = tk.Frame(scrollable_frame, bg=colors['bg_primary'])
    left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 3))

    middle_column = tk.Frame(scrollable_frame, bg=colors['bg_primary'])
    middle_column.pack(side=tk.LEFT, fill=tk.Y, padx=(3, 3))

    right_column = tk.Frame(scrollable_frame, bg=colors['bg_primary'])
    right_column.pack(side=tk.LEFT, fill=tk.Y, padx=(3, 10))

    # 为每个点创建卡片，三列分布
    for i, point in enumerate(points_data):
        # 确定卡片放在哪一列
        if i % 3 == 0:
            target_column = left_column
        elif i % 3 == 1:
            target_column = middle_column
        else:
            target_column = right_column

        # 卡片容器
        card = tk.Frame(target_column, bg=colors['bg_secondary'], relief='raised', bd=2)
        card.pack(fill=tk.X, pady=8)

        # 卡片内部容器 - 增加内边距
        card_inner = tk.Frame(card, bg=colors['bg_secondary'])
        card_inner.pack(fill=tk.X, padx=15, pady=15)

        # 顶部：点编号
        top_section = tk.Frame(card_inner, bg=colors['bg_secondary'])
        top_section.pack(fill=tk.X, pady=(0, 8))

        # 点编号标签 - 更大
        point_num_label = tk.Label(top_section, text=f"#{i+1}",
                                   bg=colors['accent_blue'], fg='white',
                                   font=('Segoe UI', 22, 'bold'),
                                   width=3, height=1)
        point_num_label.pack(side=tk.LEFT, padx=(0, 10))

        # 颜色预览方块 - 更大
        color = point['rgb']
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
        color_preview = tk.Canvas(top_section, width=55, height=55,
                                 bg=color_hex, highlightthickness=2,
                                 highlightbackground=colors['border'])
        color_preview.pack(side=tk.LEFT)

        # 分隔线
        separator = tk.Frame(card_inner, bg=colors['border'], height=1)
        separator.pack(fill=tk.X, pady=(8, 8))

        # 底部：详细信息
        info_section = tk.Frame(card_inner, bg=colors['bg_secondary'])
        info_section.pack(fill=tk.X)

        # RGB
        row0 = tk.Frame(info_section, bg=colors['bg_secondary'])
        row0.pack(fill=tk.X, pady=3)
        tk.Label(row0, text="RGB:", bg=colors['bg_secondary'],
                fg=colors['accent_green'], font=('Segoe UI', 10, 'bold'),
                width=7, anchor='w').pack(side=tk.LEFT)
        tk.Label(row0, text=f"({color[0]}, {color[1]}, {color[2]})",
                bg=colors['bg_secondary'], fg=colors['text_primary'],
                font=('Consolas', 10)).pack(side=tk.LEFT)

        # 位置
        row1 = tk.Frame(info_section, bg=colors['bg_secondary'])
        row1.pack(fill=tk.X, pady=3)
        tk.Label(row1, text="位置:", bg=colors['bg_secondary'],
                fg=colors['accent_green'], font=('Segoe UI', 10, 'bold'),
                width=7, anchor='w').pack(side=tk.LEFT)
        tk.Label(row1, text=f"({point['x']}, {point['y']})",
                bg=colors['bg_secondary'], fg=colors['text_primary'],
                font=('Consolas', 10)).pack(side=tk.LEFT)

        # CIE xy
        row2 = tk.Frame(info_section, bg=colors['bg_secondary'])
        row2.pack(fill=tk.X, pady=3)
        tk.Label(row2, text="CIE xy:", bg=colors['bg_secondary'],
                fg=colors['accent_green'], font=('Segoe UI', 10, 'bold'),
                width=7, anchor='w').pack(side=tk.LEFT)
        tk.Label(row2, text=f"({point['cie1931'][0]:.2f}, {point['cie1931'][1]:.2f})",
                bg=colors['bg_secondary'], fg=colors['text_primary'],
                font=('Consolas', 10)).pack(side=tk.LEFT)

        # CIE u'v'
        row3 = tk.Frame(info_section, bg=colors['bg_secondary'])
        row3.pack(fill=tk.X, pady=3)
        tk.Label(row3, text="u'v':", bg=colors['bg_secondary'],
                fg=colors['accent_green'], font=('Segoe UI', 10, 'bold'),
                width=7, anchor='w').pack(side=tk.LEFT)
        tk.Label(row3, text=f"({point['uv'][0]:.3f}, {point['uv'][1]:.3f})",
                bg=colors['bg_secondary'], fg=colors['text_primary'],
                font=('Consolas', 10)).pack(side=tk.LEFT)

        # CIELAB
        row4 = tk.Frame(info_section, bg=colors['bg_secondary'])
        row4.pack(fill=tk.X, pady=3)
        tk.Label(row4, text="LAB:", bg=colors['bg_secondary'],
                fg=colors['accent_green'], font=('Segoe UI', 10, 'bold'),
                width=7, anchor='w').pack(side=tk.LEFT)
        tk.Label(row4, text=f"L={point['lab'][0]:.1f} a={point['lab'][1]:.1f} b={point['lab'][2]:.1f}",
                bg=colors['bg_secondary'], fg=colors['text_primary'],
                font=('Consolas', 10)).pack(side=tk.LEFT)

    # 布局画布和滚动条
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 底部按钮栏
    btn_frame = tk.Frame(info_window, bg=colors['bg_primary'])
    btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    # 窗口关闭函数（先定义，供关闭按钮使用）
    def on_close():
        info_window.unbind("<MouseWheel>")
        info_window.destroy()

    # 关闭按钮
    close_btn = tk.Button(btn_frame, text="关闭",
                         command=on_close,
                         bg=colors['accent_blue'], fg=colors['text_primary'],
                         font=('Segoe UI', 11, 'bold'), relief='flat',
                         padx=30, pady=10, cursor='hand2')
    close_btn.pack(side=tk.RIGHT)

    # 鼠标悬停效果
    def on_enter(e):
        close_btn.config(bg='#3498db')
    def on_leave(e):
        close_btn.config(bg=colors['accent_blue'])
    close_btn.bind("<Enter>", on_enter)
    close_btn.bind("<Leave>", on_leave)

    # 使窗口居中
    info_window.update_idletasks()
    x = (info_window.winfo_screenwidth() - 850) // 2
    y = (info_window.winfo_screenheight() - 850) // 2
    info_window.geometry(f"850x850+{x}+{y}")

    # 启用鼠标滚轮滚动 - 只绑定到窗口，避免全局绑定
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # 绑定到窗口而不是全局
    info_window.bind("<MouseWheel>", on_mousewheel)

    # 设置窗口关闭协议
    info_window.protocol("WM_DELETE_WINDOW", on_close)

    info_window.focus_set()
    info_window.grab_set()

def show_combined_plot(points_data):
    """显示色度图"""
    if not points_data:
        messagebox.showwarning("无数据", "请先选择至少一个点")
        return
    plot_combined_chromaticity_diagrams(points_data)

def show_lab_3d_plot(points_data):
    """显示LAB 3D色度图"""
    if not points_data:
        messagebox.showwarning("无数据", "请先选择至少一个点")
        return
    plot_lab_3d_colorspace(points_data)

def save_lab_3d_plot(points_data):
    """保存LAB 3D色度图"""
    if not points_data:
        messagebox.showwarning("无数据", "没有可保存的点数据")
        return

    filepath = save_file_dialog([("PNG文件", "*.png"), ("PDF文件", "*.pdf"), ("所有文件", "*.*")],
                               generate_filename("lab_3d_colorspace"), "保存LAB 3D色度图")
    if filepath:
        plot_lab_3d_colorspace(points_data, save_path=filepath)
        messagebox.showinfo("保存成功", f"LAB 3D色度图已保存到:\n{filepath}")


def get_window_size(img_width, img_height, screen_width, screen_height, margin=100):
    """计算适合屏幕的窗口尺寸"""
    window_width = min(img_width + 50, screen_width - margin)
    window_height = min(img_height + 150, screen_height - margin)
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    return window_width, window_height, x, y

def on_closing(root, points_data):
    """处理窗口关闭事件"""
    plt.close('all')
    if points_data:
        messagebox.showinfo("退出", f"已分析 {len(points_data)} 个点，程序即将退出。")
    else:
        messagebox.showinfo("退出", "程序即将退出。")
    root.destroy()
    root.quit()
    sys.exit(0)


def display_image(img):
    """显示图片并绑定点击事件"""
    root = tk.Tk()
    root.title("多点颜色分析工具")
    colors = setup_modern_ui()
    root.configure(bg=colors['bg_primary'])

    # 设置窗口大小
    img_width, img_height = img.size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width, window_height, x, y = get_window_size(img_width, img_height, screen_width, screen_height, margin=50)
    window_width = max(window_width, 1200)  # 减少最小宽度限制
    window_height = max(window_height, 1100)  # 增加高度，使点分析和色彩调色板都有更多空间

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.minsize(1000, 1000)  # 增加最小窗口高度

    points_data = []
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, points_data))

    # 主容器
    main_container = tk.Frame(root, bg=colors['bg_primary'])
    main_container.pack(fill=tk.BOTH, expand=True)

    # 标题栏
    title_bar = tk.Frame(main_container, bg=colors['bg_secondary'], height=60)
    title_bar.pack(fill=tk.X, padx=10, pady=(10, 5))
    title_bar.pack_propagate(False)

    tk.Label(title_bar, text="多点颜色分析工具", bg=colors['bg_secondary'],
            fg=colors['text_primary'], font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT, padx=20, pady=15)
    tk.Label(title_bar, text="点击图片获取颜色坐标信息", bg=colors['bg_secondary'],
            fg=colors['text_secondary'], font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 20), pady=20)

    # 内容区域 - 两栏布局：左侧图像，右侧上下两栏
    content_frame = tk.Frame(main_container, bg=colors['bg_primary'])
    content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))


    left_frame = tk.Frame(content_frame, bg=colors['bg_secondary'], relief='ridge', bd=1)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

    right_container = tk.Frame(content_frame, bg=colors['bg_secondary'], relief='ridge', bd=1, width=600)
    right_container.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))  # 只填充垂直方向
    right_container.pack_propagate(False)
    # 强制设置固定宽度，防止内容扩展
    right_container.grid_propagate(False)

    # 右侧上下两栏 - 调整调色板空间，确保9-10个点时不会重新布局
    # 每个卡片约100px高度，5个卡片需要约500px，加上标题等约需580px
    right_container.grid_rowconfigure(0, weight=2, minsize=350)  # 点分析区域稍微减少
    right_container.grid_rowconfigure(1, weight=3, minsize=580)  # 调色板区域增加空间

    # 创建框架 - 简洁的布局
    info_frame = tk.Frame(right_container, bg=colors['bg_secondary'])
    info_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 2))

    # 调色板区域
    palette_frame = tk.Frame(right_container, bg=colors['bg_secondary'])
    palette_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(2, 5))

    # 创建组件
    image_viewer = ImageViewer(left_frame)
    image_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    image_viewer.set_image(img)

    info_panel = create_info_panel_modern(info_frame, points_data, colors)
    color_palette_updater = create_color_palette_modern(palette_frame, points_data, colors)

    toolbar = create_toolbar_modern(main_container, root, points_data, image_viewer, info_panel, color_palette_updater)
    toolbar.pack(fill=tk.X, padx=10, pady=5)

    image_viewer.click_callback = lambda x, y: on_click(x, y, image_viewer, points_data, info_panel, color_palette_updater)
    update_recent_points_info(points_data, info_panel['recent_points_text'])

    # 初始化色彩调色盘的提示显示
    color_palette_updater()

    # 绑定快捷键
    # Ctrl + 鼠标滚轮缩放
    def ctrl_mousewheel(event):
        if event.delta > 0:
            image_viewer.zoom_in()
        else:
            image_viewer.zoom_out()

    # 方向键控制滚动
    def arrow_up(event):
        image_viewer.canvas.yview_scroll(-1, "units")

    def arrow_down(event):
        image_viewer.canvas.yview_scroll(1, "units")

    def arrow_left(event):
        image_viewer.canvas.xview_scroll(-1, "units")

    def arrow_right(event):
        image_viewer.canvas.xview_scroll(1, "units")

    # Ctrl + 鼠标滚轮缩放
    root.bind('<Control-MouseWheel>', ctrl_mousewheel)

    # 方向键控制滚动
    root.bind('<Up>', arrow_up)
    root.bind('<Down>', arrow_down)
    root.bind('<Left>', arrow_left)
    root.bind('<Right>', arrow_right)

    # 强制设置右侧容器宽度确保生效
    root.update_idletasks()
    right_container.config(width=600)
    # 确保宽度固定，添加额外的约束
    right_container.pack_forget()
    right_container.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
    right_container.config(width=600)

    root.mainloop()


def open_image():
    """打开文件选择对话框"""
    filepath = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if filepath:
        img = Image.open(filepath)
        display_image(img)

if __name__ == "__main__":
    open_image()