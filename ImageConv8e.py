# 2026-05-13 ~ 19 ImageConv8e (CC) YC Shaw 2026
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# Import Matplotlib libraries for embedding in Tkinter
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class StarAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Star Analysis Workstation")
        self.root.geometry("1100x820")

        # Core image and layout variables
        self.cv_original = None
        self.cv_current = None
        self.tk_display = None
        self.display_ratio = 1.0
        self.img_offset = (0, 0)

        self._create_menu()
        self._setup_ui()

    def _create_menu(self):
        """Create the main window menu bar in English"""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open File", command=self.load_image)
        file_menu.add_command(label="Save File", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Reset Image", command=self.reset_image)

        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About Software",
                               command=lambda: messagebox.showinfo("About", "Star Enhance Tool v8\n(CC) YC Shaw 2026\nhelloshiau@gmail.com"))

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        menubar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menubar)

    def _setup_ui(self):
        """Configure the English UI layout layout according to requirements"""
        # 1. Load Button
        tk.Button(self.root, text="Select Image File", command=self.load_image, width=20, height=2).pack(pady=10)

        # 2. Path Entry Box
        self.path_entry = tk.Entry(self.root, width=110, justify='center')
        self.path_entry.pack(pady=5)

        # 3. Dual Canvas Section (Horizontal alignment via Frame)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10)

        # Canvas1: Image Display Area (Strictly 600x450)
        self.canvas1 = tk.Canvas(canvas_frame, width=600, height=450, bg="#1e1e1e")
        self.canvas1.pack(side=tk.LEFT, padx=10)
        self.canvas1.bind("<Button-1>", self.get_pixel_data)

        # Canvas2: Histogram Container (Strictly 300x225)
        self.hist_frame = tk.Frame(canvas_frame, width=300, height=225)
        self.hist_frame.pack(side=tk.LEFT, padx=10)
        self.hist_frame.pack_propagate(False)  # Lock size container

        # Initialize Matplotlib Figure
        self.fig = Figure(figsize=(3, 2.25), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. Six Feature Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        filters = [
            ("Negative", self.apply_negative), ("Sharp", self.apply_sharp),
            ("Edge", self.apply_edge), ("X2", self.apply_x2),
            ("to Gray", self.apply_gray), ("Star Enhance", self.apply_star_enhance)
        ]
        for text, cmd in filters:
            tk.Button(btn_frame, text=text, width=13, height=2, command=cmd).pack(side=tk.LEFT, padx=4)

        # 5. Progress Bar
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(pady=5, fill=tk.X, padx=80)
        self.progress_label = tk.Label(progress_frame, text="Star Enhance Progress: 0%", font=("Arial", 10))
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=600, mode="determinate")
        self.progress_bar.pack(pady=2)

        # 6. Pixel Data Row (Ensures inline text data alignment)
        info_line_frame = tk.Frame(self.root)
        info_line_frame.pack(pady=10)

        # Left fixed prompt text
        tk.Label(info_line_frame, text="Click Mouse Left Button for Pixel Data : ", font=("Arial", 11, "bold"),
                 fg="blue").pack(side=tk.LEFT)
        # Right dynamic data label
        self.pixel_info_label = tk.Label(info_line_frame, text="Click on the image to view data",
                                         font=("Courier", 12, "bold"), fg="#333333")
        self.pixel_info_label.pack(side=tk.LEFT, padx=5)

        # 7. Save to Another Directory Button
        tk.Button(self.root, text="Save Image to Another Directory", command=self.save_image,
                  bg="#28a745", fg="white", width=35, height=2).pack(pady=15)

    def load_image(self):
        """Read the image file and initialize parameters"""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

            # Using numpy buffer to decode safely with Windows file paths
            raw_data = np.fromfile(path, dtype=np.uint8)
            self.cv_original = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            self.cv_current = self.cv_original.copy()
            self._update_interface()

    def _update_interface(self):
        """Refresh image canvas and histogram graphs"""
        self._render_image_canvas()
        self._render_histogram()

    def _render_image_canvas(self):
        """Scale and render OpenCV image onto Canvas1 center (600x450)"""
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
        """Calculate image distribution and plot to Canvas2 (300x225)"""
        if self.cv_current is not None:
            self.ax.clear()
            self.ax.set_xlim([0, 255])
            self.ax.get_yaxis().set_visible(False)
            self.ax.tick_params(labelsize=8)
            self.ax.set_title("R G B Histogram (0-255)", fontsize=8)

            if len(self.cv_current.shape) == 2:
                hist = cv2.calcHist([self.cv_current], [0], None, [256], [0, 256])
                self.ax.plot(hist, color='gray', linewidth=1.5)
                self.ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.2)
            else:
                colors = (('b', 'blue'), ('g', 'green'), ('r', 'red'))
                for i, (col_code, col_name) in enumerate(colors):
                    hist = cv2.calcHist([self.cv_current], [i], None, [256], [0, 256])
                    self.ax.plot(hist, color=col_name, linewidth=1.2, alpha=0.8)

            self.fig.tight_layout()
            self.hist_canvas.draw()

    def get_pixel_data(self, event):
        """Map mouse clicks to original matrix coordinates and render data"""
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

    def reset_image(self):
        if self.cv_original is not None:
            self.cv_current = self.cv_original.copy()
            self._update_interface()
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Star Enhance Progress: 0%")

    # --- Six Image Processing Filters ---
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

    def apply_star_enhance(self):
        """Enhance stars using Thresholded Center of Mass and report progress"""
        if self.cv_current is None: return

        is_color = (len(self.cv_current.shape) == 3)
        working_gray = cv2.cvtColor(self.cv_current, cv2.COLOR_BGR2GRAY) if is_color else self.cv_current.copy()

        _, thresholded = cv2.threshold(working_gray, 200, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded)

        total_stars = num_labels - 1
        if total_stars <= 0:
            messagebox.showinfo("Notice", "No high-brightness star targets detected above threshold.")
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
            percentage = int((i / total_stars) * 100)
            self.progress_label.config(text=f"Star Enhance Progress: {percentage}%")

            if i % 10 == 0 or i == total_stars:
                self.root.update_idletasks()

        boost_mask = cv2.GaussianBlur(boost_mask, (3, 3), 0)

        if is_color:
            for ch in range(3):
                enhanced_channel = self.cv_current[:, :, ch].astype(np.uint16) + (boost_mask.astype(np.uint16) // 2)
                self.cv_current[:, :, ch] = np.clip(enhanced_channel, 0, 255).astype(np.uint8)
        else:
            enhanced_gray = self.cv_current.astype(np.uint16) + (boost_mask.astype(np.uint16) // 2)
            self.cv_current = np.clip(enhanced_gray, 0, 255).astype(np.uint8)

        self._update_interface()
        messagebox.showinfo("Success", "Star enhancement and histogram updated successfully!")

    def save_image(self):
        """Save image to directory via file dialog"""
        if self.cv_current is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
            if save_path:
                ext = os.path.splitext(save_path)[1]
                _, buf = cv2.imencode(ext, self.cv_current)
                buf.tofile(save_path)
                messagebox.showinfo("Success", "The image file has been saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = StarAnalyzerApp(root)
    root.mainloop()
