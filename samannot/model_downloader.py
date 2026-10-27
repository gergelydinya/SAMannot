"""SAM 2 Model Weights Downloader GUI and utilities."""

import os
import ssl
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import urllib.request

SAM2P1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

MODELS = {
    "SAM 2.1 Large (Recommended)": {
        "filename": "sam2.1_hiera_large.pt",
        "url": f"{SAM2P1_BASE_URL}/sam2.1_hiera_large.pt",
        "size_mb": 856.5,
        "default": True,
    },
    "SAM 2.1 Base+": {
        "filename": "sam2.1_hiera_base_plus.pt",
        "url": f"{SAM2P1_BASE_URL}/sam2.1_hiera_base_plus.pt",
        "size_mb": 308.5,
        "default": False,
    },
    "SAM 2.1 Small": {
        "filename": "sam2.1_hiera_small.pt",
        "url": f"{SAM2P1_BASE_URL}/sam2.1_hiera_small.pt",
        "size_mb": 177.3,
        "default": False,
    },
    "SAM 2.1 Tiny": {
        "filename": "sam2.1_hiera_tiny.pt",
        "url": f"{SAM2P1_BASE_URL}/sam2.1_hiera_tiny.pt",
        "size_mb": 143.6,
        "default": False,
    },
}


def get_checkpoint_dir() -> Path:
    """Return default directory for checkpoints (local checkpoints/ or user cache)."""
    local_dir = Path.cwd() / "checkpoints"
    if local_dir.is_dir():
        return local_dir.resolve()

    repo_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    if repo_dir.is_dir():
        return repo_dir.resolve()

    cache_dir = Path.home() / ".cache" / "samannot" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir.resolve()


def download_file(url: str, dest_path: Path, progress_callback=None, cancel_event: threading.Event = None) -> bool:
    """Download a file with chunked progress reporting and cancellation support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")

    ctx = ssl.create_default_context()
    try:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    except Exception:
        pass

    req = urllib.request.Request(url, headers={"User-Agent": "SAMannot-Downloader/1.0"})

    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            start_time = time.time()
            chunk_size = 1024 * 1024  # 1 MB chunk

            with open(temp_path, "wb") as f:
                while True:
                    if cancel_event and cancel_event.is_set():
                        f.close()
                        if temp_path.exists():
                            temp_path.unlink()
                        return False

                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0

                    if progress_callback:
                        progress_callback(downloaded, total_size, speed)

        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return True
        return False
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise e


class ModelDownloaderWindow(tk.Toplevel):
    """GUI Window for downloading SAM 2 model weights."""

    def __init__(self, parent=None, on_complete_callback=None):
        super().__init__(parent)
        self.parent = parent
        self.on_complete_callback = on_complete_callback

        self.title("Download SAM 2 Model Weights")
        self.geometry("620x460")
        self.minsize(560, 420)
        self.configure(bg="#f5f6f7")

        self.cancel_event = threading.Event()
        self.download_thread = None
        self.is_downloading = False

        self.target_dir_var = tk.StringVar(value=str(get_checkpoint_dir()))
        self.selected_model_var = tk.StringVar(value="SAM 2.1 Large (Recommended)")
        self.status_var = tk.StringVar(value="Ready to download.")
        self.progress_detail_var = tk.StringVar(value="")

        self.create_widgets()
        self.refresh_model_status()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_label = ttk.Label(
            main_frame,
            text="SAM 2 Model Weights Downloader",
            font=("Segoe UI", 13, "bold"),
        )
        header_label.pack(anchor="w", pady=(0, 4))

        info_label = ttk.Label(
            main_frame,
            text="SAMannot requires model checkpoint weights to perform segmentation and tracking.\n"
                 "Select a model variant below to download it into your checkpoints directory.",
            font=("Segoe UI", 9),
            foreground="#555555",
            justify="left",
        )
        info_label.pack(anchor="w", pady=(0, 12))

        # Target directory frame
        dir_frame = ttk.LabelFrame(main_frame, text="Download Destination", padding=8)
        dir_frame.pack(fill=tk.X, pady=(0, 12))

        dir_entry = ttk.Entry(dir_frame, textvariable=self.target_dir_var)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        browse_btn = ttk.Button(dir_frame, text="Browse...", command=self.browse_target_dir)
        browse_btn.pack(side=tk.RIGHT)

        # Model selection frame
        models_frame = ttk.LabelFrame(main_frame, text="Select Model Checkpoint", padding=8)
        models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        self.model_radios = {}
        self.status_labels = {}
        row = 0
        for name, meta in MODELS.items():
            r = ttk.Radiobutton(
                models_frame,
                text=f"{name} ({meta['size_mb']:.1f} MB)",
                value=name,
                variable=self.selected_model_var,
                command=self.refresh_model_status,
            )
            r.grid(row=row, column=0, sticky="w", pady=3)

            lbl = ttk.Label(models_frame, text="", font=("Segoe UI", 9, "italic"))
            lbl.grid(row=row, column=1, sticky="e", padx=(10, 0), pady=3)
            models_frame.columnconfigure(1, weight=1)

            self.model_radios[name] = r
            self.status_labels[name] = lbl
            row += 1

        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 12))

        self.progressbar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progressbar.pack(fill=tk.X, pady=(0, 4))

        self.progress_detail_label = ttk.Label(
            progress_frame,
            textvariable=self.progress_detail_var,
            font=("Segoe UI", 9),
            foreground="#444444",
        )
        self.progress_detail_label.pack(anchor="w")

        # Bottom buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(
            btn_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            foreground="#222222",
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_download, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(6, 0))

        self.download_btn = ttk.Button(btn_frame, text="Download", command=self.start_download)
        self.download_btn.pack(side=tk.RIGHT)

    def browse_target_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.target_dir_var.get(), title="Select Checkpoint Folder")
        if chosen:
            self.target_dir_var.set(chosen)
            self.refresh_model_status()

    def refresh_model_status(self):
        target_dir = Path(self.target_dir_var.get())
        for name, meta in MODELS.items():
            dest = target_dir / meta["filename"]
            lbl = self.status_labels.get(name)
            if lbl:
                if dest.exists() and dest.stat().st_size > 1024 * 1024:
                    size_mb = dest.stat().st_size / (1024 * 1024)
                    lbl.config(text=f"Installed ({size_mb:.1f} MB)", foreground="#2e7d32")
                else:
                    lbl.config(text="Not downloaded", foreground="#c62828")

    def start_download(self):
        if self.is_downloading:
            return

        selected_name = self.selected_model_var.get()
        meta = MODELS.get(selected_name)
        if not meta:
            return

        target_dir = Path(self.target_dir_var.get())
        dest_path = target_dir / meta["filename"]

        if dest_path.exists():
            if not messagebox.askyesno(
                "Overwrite Existing Model",
                f"{meta['filename']} already exists in the destination folder.\nDo you want to re-download and overwrite it?",
                parent=self,
            ):
                return

        self.is_downloading = True
        self.cancel_event.clear()
        self.download_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Connecting to download {meta['filename']}...")
        self.progressbar["value"] = 0

        self.download_thread = threading.Thread(
            target=self._download_worker,
            args=(meta["url"], dest_path),
            daemon=True,
        )
        self.download_thread.start()

    def _download_worker(self, url: str, dest_path: Path):
        def on_progress(downloaded, total, speed):
            pct = (downloaded / total * 100) if total > 0 else 0
            dl_mb = downloaded / (1024 * 1024)
            tot_mb = total / (1024 * 1024)
            speed_mb = speed / (1024 * 1024)
            self.after(0, lambda: self._update_progress_ui(pct, dl_mb, tot_mb, speed_mb))

        try:
            success = download_file(url, dest_path, progress_callback=on_progress, cancel_event=self.cancel_event)
            if success:
                self.after(0, lambda: self._on_download_success(dest_path))
            else:
                self.after(0, lambda: self._on_download_cancelled())
        except Exception as e:
            self.after(0, lambda: self._on_download_error(str(e)))

    def _update_progress_ui(self, pct: float, dl_mb: float, tot_mb: float, speed_mb: float):
        self.progressbar["value"] = pct
        self.progress_detail_var.set(
            f"{pct:.1f}% — {dl_mb:.1f} MB / {tot_mb:.1f} MB ({speed_mb:.1f} MB/s)"
        )
        self.status_var.set("Downloading weights...")

    def _on_download_success(self, dest_path: Path):
        self.is_downloading = False
        self.download_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progressbar["value"] = 100
        self.status_var.set("Download completed successfully!")
        self.refresh_model_status()

        messagebox.showinfo(
            "Download Complete",
            f"The download is complete!\n\nModel weights saved to:\n{dest_path}",
            parent=self,
        )

        callback = self.on_complete_callback
        self.destroy()

        if callback:
            callback(str(dest_path))

    def _on_download_cancelled(self):
        self.is_downloading = False
        self.download_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progressbar["value"] = 0
        self.progress_detail_var.set("")
        self.status_var.set("Download cancelled.")
        self.refresh_model_status()

    def _on_download_error(self, err_msg: str):
        self.is_downloading = False
        self.download_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.status_var.set("Download failed.")
        messagebox.showerror("Download Error", f"Failed to download model weights:\n{err_msg}", parent=self)
        self.refresh_model_status()

    def cancel_download(self):
        if self.is_downloading:
            self.status_var.set("Cancelling download...")
            self.cancel_event.set()

    def on_close(self):
        if self.is_downloading:
            if messagebox.askyesno("Cancel Download", "A download is currently in progress. Cancel and exit?", parent=self):
                self.cancel_event.set()
                self.destroy()
        else:
            self.destroy()


def open_downloader(parent=None, on_complete_callback=None):
    """Helper function to open the ModelDownloaderWindow."""
    win = ModelDownloaderWindow(parent=parent, on_complete_callback=on_complete_callback)
    win.lift()
    win.focus_force()
    return win


def main():
    """Standalone CLI / script entry point."""
    root = tk.Tk()
    root.withdraw()
    win = ModelDownloaderWindow(parent=root, on_complete_callback=lambda p: print(f"Downloaded to {p}"))
    win.wait_window()
    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
