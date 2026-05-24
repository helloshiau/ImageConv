# 2026-05-13 v1 - v6  2026-05-18 ImageConv10e v10e
# 2026-05-24 ImageConv10 v10e (CC) YC Shaw 2026
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import mss

# 導入 Matplotlib 相關套件用於嵌入 Tkinter 視窗
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class UpdatedImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing & Star Enhancement Tools v10")

        # 依規格設定主視窗解析度為 1150x900  1100X900
        self.root.geometry("1100x900")
        #self.root.resizable(False, False)
        # 允許使用者自由調整主視窗解析度
        self.root.resizable(True, True)

        # 核心影像與視窗排版變數
        self.cv_original = None
        self.cv_current = None
        self.tk_display = None
        self.tk_fft_display = None  # 用於在 Canvas2 顯示 FFT 的圖片變數
        self.display_ratio = 1.0
        self.img_offset = (0, 0)

        # 建立控制滑桿的 Tkinter 變數
        self.lower_level = tk.IntVar(value=0)
        self.upper_level = tk.IntVar(value=255)

        self._create_menu()
        self._setup_ui()

    def _create_menu(self):
        """建立主視窗繁體中文選單列"""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image File", command=self.load_image)
        file_menu.add_command(label="Save Image File", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Screen Capture", command=self.start_screen_capture)
        edit_menu.add_command(label="Restore Original Image", command=self.reset_image)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Image Processing & Star Enhancement Tools v10\n(CC) YC Shaw 2026\nhelloshiau@gmail.com"))

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        menubar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menubar)

    def _setup_ui(self):
        """配置符合規格的 UI 介面佈局"""
        # 1. 讀取按鈕
        tk.Button(self.root, text="Open an Image File", command=self.load_image, width=20, height=2).pack(pady=10)

        # 2. 路徑顯示方框 (Entry)
        self.path_entry = tk.Entry(self.root, width=120, justify='center')
        self.path_entry.pack(pady=5)

        # 3. 雙 Canvas 區塊 (使用 Frame 水平排列)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10)

        # Canvas1: 影像顯示區 (規格設定為 600x450)
        self.canvas1 = tk.Canvas(canvas_frame, width=600, height=450, bg="#1e1e1e")
        self.canvas1.pack(side=tk.LEFT, padx=15)
        self.canvas1.bind("<Button-1>", self.get_pixel_data)

        # Canvas2: 直方圖/FFT顯示區容器 (規格調大為 400x300)
        self.canvas2_container = tk.Frame(canvas_frame, width=400, height=300, bg="#1e1e1e")
        self.canvas2_container.pack(side=tk.LEFT, padx=15)
        self.canvas2_container.pack_propagate(False)  # 鎖定框架大小

        # 建立純 Tkinter Canvas2 用於未來切換顯示 FFT 影像 (規格 400x300)
        self.canvas2_img = tk.Canvas(self.canvas2_container, width=400, height=300, bg="#1e1e1e")

        # 初始化 Matplotlib 畫布並嵌入 Canvas2 容器中作為初始直方圖
        self.fig = Figure(figsize=(4, 3), dpi=100)  # 4x3 吋剛好符合 400x300 像素
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas2_container)
        self.hist_widget = self.hist_canvas.get_tk_widget()
        self.hist_widget.pack(fill=tk.BOTH, expand=True)

        # 綁定 Matplotlib 畫布的滑鼠點擊事件 (左鍵與右鍵)
        self.fig.canvas.mpl_connect('button_press_event', self.get_histogram_click)

        # 4. 八個功能按鍵 (Canvas1 正下方)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        filters = [
            ("Negative", self.apply_negative), ("Sharp", self.apply_sharp),
            ("Edge", self.apply_edge), ("X2", self.apply_x2),
            ("to Gray", self.apply_gray), ("Histogram Adjust", self.apply_histogram_adjust),
            ("Star Enhance", self.apply_star_enhance), ("FFT", self.apply_fft)
        ]
        for text, cmd in filters:
            tk.Button(btn_frame, text=text, width=14, height=2, command=cmd).pack(side=tk.LEFT, padx=4)

        # 5. 八個功能按鍵下方顯示指定的提示文字與數據
        info_line_frame = tk.Frame(self.root)
        info_line_frame.pack(pady=5)
        tk.Label(info_line_frame, text="On Image : Click Mouse Left Button for Pixel Data : ", font=("Arial", 10, "bold"),
                 fg="blue").pack(side=tk.LEFT)
        self.pixel_info_label = tk.Label(info_line_frame, text="Click image to view data", font=("Courier", 11, "bold"),
                                         fg="#333333")
        self.pixel_info_label.pack(side=tk.LEFT, padx=5)

        # 6. 再下方一行創建同一行兩個 Scale Widget
        scale_frame = tk.Frame(self.root)
        scale_frame.pack(pady=5)

        tk.Label(scale_frame, text="Left Mouse for lower level :", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.scale1 = tk.Scale(scale_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.lower_level,
                               length=240)
        self.scale1.pack(side=tk.LEFT, padx=15)

        tk.Label(scale_frame, text="Right Mouse for upper_level :", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.scale2 = tk.Scale(scale_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.upper_level,
                               length=240)
        self.scale2.pack(side=tk.LEFT, padx=15)

        # 7. 再下方一行創建一個 ProgressBar 用以顯示"Star Enhance" 的進度
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(pady=5, fill=tk.X, padx=100)
        self.progress_label = tk.Label(progress_frame, text="Star Enhance Progress: 0%", font=("Arial", 10))
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.pack(pady=2)

        # 8. 再下一行顯示指定文字："Mouse Left Button for lower level, and Mouse Right Button for upper level og Histogram : "
        hist_info_frame = tk.Frame(self.root)
        hist_info_frame.pack(pady=5)
        tk.Label(hist_info_frame,
                 text="On Histogram : Mouse Left Button for lower level, and Mouse Right Button for upper level : ",
                 font=("Arial", 10, "bold"), fg="purple").pack(side=tk.LEFT)
        self.hist_info_label = tk.Label(hist_info_frame, text="L-Click / R-Click on Canvas2",
                                        font=("Courier", 11, "bold"), fg="#333333")
        self.hist_info_label.pack(side=tk.LEFT, padx=5)

        # 9. 另存目錄之按鍵 (最下方)
        tk.Button(self.root, text="Save the processed Image to Another File", command=self.save_image,
                  bg="#28a745", fg="white", width=35, height=2).pack(pady=10)

    def load_image(self):
        """讀取影像檔案"""
        path = filedialog.askopenfilename(filetypes=[("Image File", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

            raw_data = np.fromfile(path, dtype=np.uint8)
            self.cv_original = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            self.cv_current = self.cv_original.copy()
            self._update_interface()

    def _update_interface(self, show_fft_on_c2=False):
        """重新整理畫布影像與右側直方圖數據"""
        self._render_image_canvas()

        if show_fft_on_c2:
            # 如果是 FFT 模式，隱藏直方圖小部件，顯示並重繪 Canvas2 圖片
            self.hist_widget.pack_forget()
            self.canvas2_img.pack(fill=tk.BOTH, expand=True)
        else:
            # 一般濾鏡模式，切換回顯示直方圖小部件
            self.canvas2_img.pack_forget()
            self.hist_widget.pack(fill=tk.BOTH, expand=True)
            self._render_histogram()

    def _render_image_canvas(self):
        """將影像等比例縮放至 600x450 並繪製於 Canvas1"""
        if self.cv_current is not None:
            if len(self.cv_current.shape) == 2:
                img_rgb = cv2.cvtColor(self.cv_current, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(img_rgb)
            orig_w, orig_h = pil_img.size

            pil_img.thumbnail((600, 450))
            new_w, new_h = pil_img.size
            self.display_ratio = orig_w / new_w

            self.tk_display = ImageTk.PhotoImage(pil_img)
            self.canvas1.delete("all")
            self.canvas1.create_image(300, 225, anchor=tk.CENTER, image=self.tk_display)
            self.img_offset = (300 - new_w // 2, 225 - new_h // 2)

    def _render_histogram(self):
        """計算影像亮度值(0-255)並將三通道直方圖輸出至 Canvas2"""
        if self.cv_current is not None:
            self.ax.clear()
            self.ax.set_xlim([0, 255])
            self.ax.get_yaxis().set_visible(False)
            self.ax.tick_params(labelsize=8)
            self.ax.set_title("R G B Histogram (0-255)", fontsize=8)

            if len(self.cv_current.shape) == 2:
                hist = cv2.calcHist([self.cv_current], [0], None, [256], [0, 256])
                self.ax.plot(hist, color='gray', linewidth=1.5)
            else:
                colors = (('b', 'blue'), ('g', 'green'), ('r', 'red'))
                for i, (col_code, col_name) in enumerate(colors):
                    hist = cv2.calcHist([self.cv_current], [i], None, [256], [0, 256])
                    self.ax.plot(hist, color=col_name, linewidth=1.2, alpha=0.8)

            self.fig.tight_layout()
            self.hist_canvas.draw()

    def get_pixel_data(self, event):
        """滑鼠左鍵點擊 Canvas1 影像取得像素 R, G, B 數據"""
        if self.cv_current is None: return

        ix = int((event.x - self.img_offset[0]) * self.display_ratio)
        iy = int((event.y - self.img_offset[1]) * self.display_ratio)

        h, w = self.cv_current.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            pixel = self.cv_current[iy, ix]
            if len(self.cv_current.shape) == 2:
                r = g = b = pixel
            else:
                b, g, r = pixel

            self.pixel_info_label.config(text=f"X: {ix}, Y: {iy} | R: {r}, G: {g}, B: {b}")

    def get_histogram_click(self, event):
        """處理 Canvas2 直方圖上的滑鼠點擊事件並傳值給滑桿"""
        if event.xdata is None: return

        x_val = int(np.clip(event.xdata, 0, 255))
        if event.button == 1:  # 左鍵
            self.lower_level.set(x_val)
            self.hist_info_label.config(text=f"lower_level set to: {x_val}")
        elif event.button == 3:  # 右鍵
            self.upper_level.set(x_val)
            self.hist_info_label.config(text=f"upper_level set to: {x_val}")

    def start_screen_capture(self):
        """啟動 Screen Capture 畫面擷取遮罩"""
        self.root.iconify()
        self.root.after(300, ScreenCaptureMask, self)

    def reset_image(self):
        if self.cv_original is not None:
            self.cv_current = self.cv_original.copy()
            self._update_interface()
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Star Enhance Progress: 0%")

    # --- 八大核心濾鏡演算法 ---
    def apply_negative(self):
        if self.cv_current is not None:
            self.cv_current = 255 - self.cv_current
            self._update_interface()

    def apply_sharp(self):
        if self.cv_current is not None:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            self.cv_current = cv2.filter2D(self.cv_current, -1, kernel)
            self._update_interface()

    def apply_edge(self):
        if self.cv_current is not None:
            gray = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY) if len(
                self.cv_current.shape) == 3 else self.cv_current
            self.cv_current = cv2.Canny(gray, 100, 200)
            self._update_interface()

    def apply_x2(self):
        if self.cv_current is not None:
            res = self.cv_current.astype(np.uint16) * 2
            self.cv_current = np.clip(res, 0, 255).astype(np.uint8)
            self._update_interface()

    def apply_gray(self):
        if self.cv_current is not None and len(self.cv_current.shape) == 3:
            self.cv_current = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY)
            self._update_interface()

    def apply_histogram_adjust(self):
        if self.cv_current is None: return
        low = self.lower_level.get()
        high = self.upper_level.get()
        if low >= high:
            messagebox.showerror("Error!", "lower_level must be less than upper_level！")
            return
        img_float = self.cv_current.astype(np.float32)
        adjusted = (img_float - low) * (255.0 / (high - low))
        self.cv_current = np.clip(adjusted, 0, 255).astype(np.uint8)
        self._update_interface()

    def apply_star_enhance(self):
        if self.cv_current is None: return
        is_color = (len(self.cv_current.shape) == 3)
        working_gray = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY) if is_color else self.cv_current.copy()

        _, thresholded = cv2.threshold(working_gray, 200, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded)
        total_stars = num_labels - 1
        if total_stars <= 0:
            messagebox.showinfo("Hint : ", "No Obvious Star Detected.")
            return

        self.progress_bar['maximum'] = total_stars
        self.progress_bar['value'] = 0
        boost_mask = np.zeros_like(working_gray, dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 1 <= area <= 150:
                component_mask = (labels == i)
                moments = cv2.moments(component_mask.astype(np.uint8))
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.circle(boost_mask, (cx, cy), 2, 255, -1)

            self.progress_bar['value'] = i
            self.progress_label.config(text=f"Star Enhance Progress: {int((i / total_stars) * 100)}%")
            if i % 10 == 0 or i == total_stars:
                self.root.update_idletasks()

        boost_mask = cv2.GaussianBlur(boost_mask, (3, 3), 0)
        if is_color:
            for ch in range(3):
                res = self.cv_current[:, :, ch].astype(np.uint16) + (boost_mask // 2)
                self.cv_current[:, :, ch] = np.clip(res, 0, 255).astype(np.uint8)
        else:
            res = self.cv_current.astype(np.uint16) + (boost_mask // 2)
            self.cv_current = np.clip(res, 0, 255).astype(np.uint8)
        self._update_interface()

    def apply_fft(self):
        """將 Canvas1 的影像做 FFT 轉換，之後將實數部分以 400x300 格式顯示於 Canvas2 之中"""
        if self.cv_current is None: return

        if len(self.cv_current.shape) == 3:
            gray_img = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = self.cv_current.copy()

        # 1. 執行二維快速傅立葉變換
        fft_data = np.fft.fft2(gray_img)
        fft_shift = np.fft.fftshift(fft_data)

        # 2. 提取實數部分 (Real Part)
        real_part = np.real(fft_shift)

        # 3. 採用對數動態範圍壓縮與歸一化處理，方便視覺化顯示
        real_abs = np.abs(real_part)
        log_display = np.log1p(real_abs)
        cv2.normalize(log_display, log_display, 0, 255, cv2.NORM_MINMAX)
        fft_uint8 = log_display.astype(np.uint8)

        # 4. 將實數圖像等比例縮放或調整至符合 Canvas2 的 400x300 大小
        pil_fft = Image.fromarray(fft_uint8)
        pil_fft.thumbnail((400, 300))

        # 5. 渲染至 Canvas2 專用的影像元件中
        self.tk_fft_display = ImageTk.PhotoImage(pil_fft)
        self.canvas2_img.delete("all")
        self.canvas2_img.create_image(200, 150, anchor=tk.CENTER, image=self.tk_fft_display)

        # 6. 通知介面更新：這次需要把直方圖切換為 FFT 影像圖面
        self._update_interface(show_fft_on_c2=True)

    def save_image(self):
        """將影像儲存於另一個目錄之中"""
        if self.cv_current is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG File", "*.png"), ("JPG File", "*.jpg")])
            if save_path:
                ext = os.path.splitext(save_path)[1]
                _, buf = cv2.imencode(ext, self.cv_current)
                buf.tofile(save_path)
                messagebox.showinfo("Successful!", "Processed Image Saved Already!")


class ScreenCaptureMask:
    """全螢幕擷取滑鼠拉矩形之半透明遮罩視窗"""

    def __init__(self, app_instance):
        self.app = app_instance
        self.top = tk.Toplevel()
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.4)
        self.top.config(cursor="cross")

        self.canvas = tk.Canvas(self.top, cursor="cross", bg="grey")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 修正後的寫法：使用大寫 mss.MSS() 完美替代 mss.mss() 消除警示
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            self.full_screenshot = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline="red", width=2)

    def on_drag(self, event):
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        self.top.destroy()

        x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
        y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)

        if (x2 - x1) > 5 and (y2 - y1) > 5:
            cropped_img = self.full_screenshot.crop((x1, y1, x2, y2))
            self.app.cv_original = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            self.app.cv_current = self.app.cv_original.copy()
            self.app.path_entry.delete(0, tk.END)
            self.app.path_entry.insert(0, "Screen Capture Data")
            self.app._update_interface()

        self.app.root.deiconify()


if __name__ == "__main__":
    root = tk.Tk()
    app = UpdatedImageApp(root)
    root.mainloop()
