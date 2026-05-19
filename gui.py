from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp",
 ".webp", ".tiff", ".tif", ".gif")
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEEP_CHECKPOINT_PATH = CHECKPOINT_DIR / "mobilenetv3_detector.pth"
MANUAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "manual_detector.joblib"
BG_MAIN = "white"
FG_MAIN = "black"
PANEL_BG = "#E8EEF8"


class DetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image Detector")
        self.geometry("760x580")
        self.minsize(760, 580)
        self.configure(bg=BG_MAIN)
        self.detector = None
        self.selected_path: Path | None = None
        self.preview_image_ref = None
        self.display()
        self._update_modes()

    #Used Buttonns In Replace Of A lables As they stopped displaying when project moved to the lended macOS 
    def display(self):
        root = tk.Frame(self, bg=BG_MAIN, padx=18, pady=14)
        root.pack(fill="both", expand=True)

        shell = tk.Frame(root, bg=BG_MAIN, bd=2, relief="solid", padx=16, pady=12)
        shell.pack(fill="both", expand=True)
        
        tk.Button(shell, text="AI Image Detector", bg=BG_MAIN, fg="#000000", activeforeground="#000000", activebackground=BG_MAIN,
            font=("Helvetica", 24, "bold"),
            relief="flat",
            bd=0, highlightthickness=0,
            padx=0,
            anchor="center",
            width=46,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        ).pack(anchor="center", pady=(4, 2))

        tk.Button(
            shell, text="Upload an image to check whether it was created by an AI",  bg=BG_MAIN,
            fg="#000000", activeforeground="#000000", activebackground=BG_MAIN, font=("Helvetica", 12, "bold"),
            relief="flat", bd=0,
            highlightthickness=0, padx=0, anchor="center",
            width=62,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        ).pack(anchor="center", pady=(0, 14))

        detector_row = tk.Frame(shell, bg=BG_MAIN)
        detector_row.pack(anchor="center", pady=(0, 12))

        tk.Button(detector_row, text="Detector:", bg=BG_MAIN, fg="#000000", activeforeground="#000000",
            activebackground=BG_MAIN,
            font=("Helvetica", 14, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="w",
            width=12,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        ).pack(side="left", padx=(0, 8))

        self.mode_var = tk.StringVar(value="deep")
        self.mode_menu = tk.OptionMenu(detector_row, self.mode_var, "deep")
        self.mode_menu.config(width=14, bg="white", fg=FG_MAIN, highlightthickness=2, relief="solid", bd=1)
        self.mode_menu["menu"].config(bg="white", fg=FG_MAIN)
        self.mode_menu.pack(side="left")

        actions = tk.Frame(shell, bg=BG_MAIN)
        actions.pack(fill="x", pady=(0, 10))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)

        self.upload_btn = tk.Button(
            actions, text="Upload Image",
            command=self.on_upload,
            bg="white", fg=FG_MAIN,
            bd=2,
            relief="solid", font=("Helvetica", 16, "bold"),
            padx=8,
            pady=6,
        )
        self.upload_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.analyze_btn = tk.Button(
            actions, text="Analyze",
            command=self.on_analyze,
            state="disabled", bg="white",
            fg=FG_MAIN,
            bd=2,
            relief="solid",
            font=("Helvetica", 16, "bold"),
            padx=8,
            pady=6,
        )

        self.analyze_btn.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        self.file_label = tk.Button(
            shell,
            text="'No file selected' or 'File Path'", bg=BG_MAIN,
            fg="#000000", activeforeground="#000000",
            activebackground=BG_MAIN,
            font=("Helvetica", 12, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="center",
            width=30,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        )
        self.file_label.pack(anchor="w", pady=(0, 10))

        content = tk.Frame(shell, bg=BG_MAIN)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        
        # The user gets to preview their image 
        preview_wrap = tk.Frame(content, bg=PANEL_BG, bd=3, relief="solid", highlightbackground="#4B5563")
        preview_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_wrap.configure(height=290)
        preview_wrap.pack_propagate(False)
        self.preview_label = tk.Button(
            preview_wrap,
            text="Upload an image\nto preview.", bg=PANEL_BG,
            fg="#000000",
            activeforeground="#000000", activebackground=PANEL_BG, font=("Helvetica", 12, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="center",
            width=26,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        )
        self.preview_label.pack(expand=True)
        #
        results_wrap = tk.Frame(content, bg=PANEL_BG, bd=3, relief="solid", highlightbackground="#4B5563", padx=10, pady=10)
        results_wrap.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        results_wrap.configure(height=290)
        results_wrap.pack_propagate(False)

        tk.Button(
            results_wrap,
            text="Results",
            bg=PANEL_BG, fg="#000000", activeforeground="#000000", activebackground=PANEL_BG, font=("Helvetica", 16, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="w",
            width=22,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        ).pack(anchor="w", pady=(0, 12))

        tk.Button(
            results_wrap,
            text="Verdict", bg=PANEL_BG, fg="#000000", activeforeground="#000000", activebackground=PANEL_BG,
            font=("Helvetica", 12, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=0, padx=0,
            anchor="w",
            width=12,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        ).pack(anchor="w")

        self.verdict_text = tk.Button(
            results_wrap,
            text="-",
            bg=PANEL_BG, fg="#000000", activeforeground="#000000",
            activebackground=PANEL_BG,
            font=("Helvetica", 12, "bold"),relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="w",
            width=20,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        )
        self.verdict_text.pack(anchor="w", pady=(0, 2))

        self.verdict_bar_var = tk.StringVar(value="Confidence: 0.00%")
        self.verdict_bar = tk.Button( results_wrap, text="Confidence: 0.00%",
            bg=PANEL_BG,fg="#000000",
            activeforeground="#000000",
            activebackground=PANEL_BG, font=("Helvetica", 12), relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="w",
            width=20,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        )
        self.verdict_bar.configure(textvariable=self.verdict_bar_var)
        self.verdict_bar.pack(anchor="w", pady=(2, 4))
        self.verdict_progress_var = tk.DoubleVar(value=0.0)
        self.verdict_progress = ttk.Progressbar(
            results_wrap,
            length=240,
            maximum=100.0,
            mode="determinate",
            variable=self.verdict_progress_var,
        )
        self.verdict_progress.pack(anchor="w", fill="x", pady=(0, 8))
        
        divider = tk.Frame(results_wrap, bg="#111827", height=2)
        divider.pack(fill="x", pady=(6, 10))
        tk.Button(
            results_wrap, text="Source", bg=PANEL_BG, fg="#000000", activeforeground="#000000", activebackground=PANEL_BG,
            font=("Helvetica", 12, "bold"),
            relief="flat",bd=0,
            highlightthickness=0, padx=0,anchor="w",
            width=12, command=lambda: None,
            cursor="arrow",
            takefocus=0,
        ).pack(anchor="w")

        self.source_text = tk.Button( results_wrap,
            text="-", bg=PANEL_BG,
            fg="#000000", activeforeground="#000000", activebackground=PANEL_BG,
            font=("Helvetica", 12, "bold"), relief="flat", bd=0,
            highlightthickness=0,padx=0,
            anchor="w",
            width=20,
            command=lambda: None,cursor="arrow",
            takefocus=0,
        )
        self.source_text.pack(anchor="w", pady=(0, 2))

        self.source_bar_var = tk.StringVar(value="Confidence: 0.00%")
        self.source_bar = tk.Button(
            results_wrap, text="Confidence: 0.00%",
            bg=PANEL_BG,fg="#000000", activeforeground="#000000", activebackground=PANEL_BG,
            font=("Helvetica", 12),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=0,
            anchor="w",
            width=20,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        )
        self.source_bar.configure(textvariable=self.source_bar_var)
        self.source_bar.pack(anchor="w", pady=(2, 4))
        self.source_progress_var = tk.DoubleVar(value=0.0)
        self.source_progress = ttk.Progressbar(
            results_wrap,
            length=240,
            maximum=100.0,
            mode="determinate",
            variable=self.source_progress_var,
        )
        self.source_progress.pack(anchor="w", fill="x", pady=(0, 0))
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Button(
            shell,
            text="Analysis Complete.",
            bg=BG_MAIN,
            fg="#000000", activeforeground="#000000", activebackground=BG_MAIN,
            font=("Helvetica", 14), relief="flat", bd=0, highlightthickness=0, padx=0,
            anchor="center", width=46,
            command=lambda: None,
            cursor="arrow",
            takefocus=0,
        )
        self.status_label.configure(textvariable=self.status_var)
        self.status_label.pack(anchor="center", pady=(10, 0))
  

    def _update_modes(self):
        availability = {
            "deep": DEEP_CHECKPOINT_PATH.exists(), "manual": MANUAL_CHECKPOINT_PATH.exists(),"hybrid": DEEP_CHECKPOINT_PATH.exists() and MANUAL_CHECKPOINT_PATH.exists(),
        }
        modes = [mode for mode, available in availability.items() if available]
        menu = self.mode_menu["menu"]
        menu.delete(0, "end")

        def _set_mode(selected_mode: str):
            self.mode_var.set(selected_mode)

        if not modes:
            modes = ["deep", "manual", "hybrid"]
            self.mode_var.set("deep")
            self.status_var.set("Checkpoints aren't found yet.")
        else:
            if self.mode_var.get() not in modes:
                self.mode_var.set(modes[0])
            self.status_var.set("Ready.")
        for mode in modes:
            menu.add_command(label=mode, command=lambda m=mode: _set_mode(m))
    
    #Makes sure the user only uploads images that are allowed 
    def on_upload(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif *.gif")],
        )
        if not file_path:
            return

        if not file_path.lower().endswith(SUPPORTED_EXTENSIONS):
            messagebox.showerror("Invalid file", "Unsupported image format.")
            return

        self.selected_path = Path(file_path)
        self.file_label.config(text=f"File: {self.selected_path.name}")
        self.analyze_btn.config(state="normal")
        self._show_preview(self.selected_path)
        self.status_var.set("Image loaded. Click Analyze.")
    
    #shows the user their image 
    def _show_preview(self, path: Path):
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 260))
        tk_img = ImageTk.PhotoImage(image)
        self.preview_image_ref = tk_img
        self.preview_label.config(image=tk_img, text="")

    def on_analyze(self):
        if self.selected_path is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        self.upload_btn.config(state="disabled")
        self.analyze_btn.config(state="disabled")
        self.status_var.set("Running Analysis...")
        self.verdict_bar_var.set("Confidence: 0.00%")
        self.source_bar_var.set("Confidence: 0.00%")
        self.verdict_progress_var.set(0.0)
        self.source_progress_var.set(0.0)

        thread = threading.Thread(target=self._run_inference, daemon=True)
        thread.start()

    def _run_inference(self):
        try:
            if self.detector is None:
                from inference import Detector

                self.detector = Detector()
            mode = self.mode_var.get().strip()
            image = Image.open(self.selected_path).convert("RGB")
            result = self.detector.predict(image, mode=mode)
            self.after(0, lambda: self._update_results(result))
        except Exception as exc:
            self.after(0, lambda: self._handle_error(exc))
   #Changes results based on iff user switches between modes
    def _update_results(self, result):
        self.verdict_text.config(text=f"{result.verdict_label}")
        self.verdict_bar_var.set(f"Confidence: {result.verdict_confidence:.2f}%")
        self.verdict_progress_var.set(min(100.0, max(0.0, float(result.verdict_confidence))))
        self.source_text.config(text=f"{result.source_label}")
        self.source_bar_var.set(f"Confidence: {result.source_confidence:.2f}%")
        self.source_progress_var.set(min(100.0, max(0.0, float(result.source_confidence))))
        self.status_var.set("Analysis complete.")
        self.upload_btn.config(state="normal")
        self.analyze_btn.config(state="normal")

    def _handle_error(self, exc: BaseException):
        self.status_var.set("Analysis failed.")
        self.verdict_progress_var.set(0.0)
        self.source_progress_var.set(0.0)
        self.upload_btn.config(state="normal")
        self.analyze_btn.config(state="normal")
        msg = str(exc).strip() or repr(exc)
        messagebox.showerror("There's Inference error", msg)


if __name__ == "__main__":
    app = DetectorApp()
    app.mainloop()
