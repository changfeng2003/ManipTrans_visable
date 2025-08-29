import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FrameComparator:
    def __init__(self, root):
        self.root = root
        self.root.title("视频帧对比工具（固定尺寸修复版）")
        self.root.geometry("1200x800")

        # 基础参数
        self.folder1 = ""
        self.folder2 = ""
        self.img_paths1 = []
        self.img_paths2 = []
        self.current_idx = 0
        self.interval = 2
        self.timer_id = None
        self.is_playing = False

        # 新增：限制最大缩放比例（避免图片过大）
        self.max_scale = 1.5  # 最大缩放为原图的1.5倍
        # 初始显示区域尺寸
        self.display_width = 500
        self.display_height = 600

        self._create_widgets()

    def _create_widgets(self):
        # 顶部控制区（同之前）
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        tk.Button(control_frame, text="选择文件夹1", command=self._select_folder1).grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="选择文件夹2", command=self._select_folder2).grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="切换间隔(秒):").grid(row=0, column=2, padx=5)
        self.interval_entry = tk.Entry(control_frame, width=5)
        self.interval_entry.insert(0, str(self.interval))
        self.interval_entry.grid(row=0, column=3, padx=5)
        tk.Button(control_frame, text="应用间隔", command=self._apply_interval).grid(row=0, column=4, padx=5)

        self.play_btn = tk.Button(control_frame, text="开始自动切换", command=self._toggle_play)
        self.play_btn.grid(row=0, column=5, padx=5)

        tk.Button(control_frame, text="上一张", command=self._prev_frame).grid(row=0, column=6, padx=5)
        tk.Button(control_frame, text="下一张", command=self._next_frame).grid(row=0, column=7, padx=5)

        self.status_label = tk.Label(control_frame, text="未加载图片")
        self.status_label.grid(row=0, column=8, padx=20)

        # 中间图片显示区（强制左右尺寸一致）
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左右标签使用相同的固定尺寸
        self.left_label = tk.Label(self.img_frame, text="请选择文件夹1", bg="#f0f0f0")
        self.left_label.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        self.right_label = tk.Label(self.img_frame, text="请选择文件夹2", bg="#f0f0f0")
        self.right_label.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)

        # 绑定窗口大小变化事件（优化版）
        self.root.bind("<Configure>", self._on_window_resize)

    def _on_window_resize(self, event):
        """优化窗口 resize 逻辑：限制最大显示尺寸"""
        if event.widget == self.root and (event.width > 500 and event.height > 500):  # 避免窗口过小
            # 计算最大可用尺寸（减去边距）
            max_available_width = (event.width - 100) // 2  # 左右各分一半
            max_available_height = event.height - 150  # 减去控制区高度

            # 限制显示尺寸不超过初始值的2倍（避免无限增大）
            self.display_width = min(max_available_width, 1000)  # 最大宽度1000
            self.display_height = min(max_available_height, 800)  # 最大高度800

            # 刷新图片
            if self.img_paths1 and self.img_paths2:
                self._load_current_frame()

    def _resize_img(self, img_path):
        """优化缩放逻辑：限制最大缩放比例，确保左右一致"""
        try:
            img = Image.open(img_path)
            original_width, original_height = img.width, img.height

            # 计算缩放比例（同时考虑显示区域和最大缩放倍数）
            # 1. 基于显示区域的比例
            w_ratio = self.display_width / original_width
            h_ratio = self.display_height / original_height
            area_ratio = min(w_ratio, h_ratio)

            # 2. 限制最大缩放（不超过原图的max_scale倍）
            scale_ratio = min(area_ratio, self.max_scale)

            # 计算最终尺寸
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)

            # 缩放图片
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 用白色背景居中填充（确保左右背景大小一致）
            bg = Image.new('RGB', (self.display_width, self.display_height), (255, 255, 255))
            x = (self.display_width - new_width) // 2
            y = (self.display_height - new_height) // 2
            bg.paste(resized_img, (x, y))
            return bg
        except Exception as e:
            print(f"加载图片失败：{img_path}，错误：{e}")
            return Image.new("RGB", (self.display_width, self.display_height), color="#ffcccc")

    # 以下函数与之前相同（省略，保持功能不变）
    def _select_folder1(self):
        self.folder1 = filedialog.askdirectory(title="选择文件夹1")
        if self.folder1:
            self.img_paths1 = self._get_sorted_img_paths(self.folder1)
            self._update_status()
            if self.folder2:
                self.current_idx = 0
                self._load_current_frame()

    def _select_folder2(self):
        self.folder2 = filedialog.askdirectory(title="选择文件夹2")
        if self.folder2:
            self.img_paths2 = self._get_sorted_img_paths(self.folder2)
            self._update_status()
            if self.folder1:
                self.current_idx = 0
                self._load_current_frame()

    def _get_sorted_img_paths(self, folder):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        img_paths = []
        for file in os.listdir(folder):
            if os.path.splitext(file)[1].lower() in img_extensions:
                img_paths.append(os.path.join(folder, file))
        img_paths.sort(key=lambda x: os.path.basename(x))
        return img_paths

    def _apply_interval(self):
        try:
            self.interval = float(self.interval_entry.get())
            if self.interval <= 0:
                messagebox.showerror("错误", "间隔需大于0")
            else:
                messagebox.showinfo("提示", f"切换间隔已设为 {self.interval} 秒")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def _toggle_play(self):
        if not self.img_paths1 or not self.img_paths2:
            messagebox.showwarning("提示", "请先选择两个文件夹")
            return

        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="开始自动切换")
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
        else:
            self.is_playing = True
            self.play_btn.config(text="暂停自动切换")
            self._auto_next_frame()

    def _auto_next_frame(self):
        if not self.is_playing:
            return
        self._next_frame()
        self.timer_id = self.root.after(int(self.interval * 1000), self._auto_next_frame)

    def _prev_frame(self):
        if not self.img_paths1 or not self.img_paths2:
            return
        self.current_idx = (self.current_idx - 1) % min(len(self.img_paths1), len(self.img_paths2))
        self._load_current_frame()

    def _next_frame(self):
        if not self.img_paths1 or not self.img_paths2:
            return
        self.current_idx = (self.current_idx + 1) % min(len(self.img_paths1), len(self.img_paths2))
        self._load_current_frame()

    def _load_current_frame(self):
        max_idx = min(len(self.img_paths1), len(self.img_paths2)) - 1
        if self.current_idx > max_idx:
            self.current_idx = 0

        left_path = self.img_paths1[self.current_idx] if self.current_idx < len(self.img_paths1) else None
        right_path = self.img_paths2[self.current_idx] if self.current_idx < len(self.img_paths2) else None

        if left_path:
            left_img = self._resize_img(left_path)
            self.left_tkimg = ImageTk.PhotoImage(left_img)
            self.left_label.config(image=self.left_tkimg, text="")
        else:
            self.left_label.config(image="", text="无此图片")

        if right_path:
            right_img = self._resize_img(right_path)
            self.right_tkimg = ImageTk.PhotoImage(right_img)
            self.right_label.config(image=self.right_tkimg, text="")
        else:
            self.right_label.config(image="", text="无此图片")

        self._update_status()

    def _update_status(self):
        if not self.img_paths1 or not self.img_paths2:
            self.status_label.config(text=f"文件夹1：{len(self.img_paths1)}张 | 文件夹2：{len(self.img_paths2)}张")
        else:
            total = min(len(self.img_paths1), len(self.img_paths2))
            self.status_label.config(text=f"进度：{self.current_idx + 1}/{total} | 文件夹1：{len(self.img_paths1)}张 | 文件夹2：{len(self.img_paths2)}张")

if __name__ == "__main__":
    root = tk.Tk()
    app = FrameComparator(root)
    root.mainloop()
    