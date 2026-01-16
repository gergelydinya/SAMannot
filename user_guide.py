import tkinter as tk
from tkinter import ttk
class UserGuideWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title("User Guide")
        self.geometry("900x600")
        self.minsize(600, 400)
        self.configure(bg="#202331")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style(self)
        style.configure(
            "Guide.TFrame",
            background="#202331",
        )
        style.configure(
            "GuideHeader.TLabel",
            background="#202331",
            foreground="#FFFFFF",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "GuideSubHeader.TLabel",
            background="#202331",
            foreground="#A0A4C0",
            font=("Segoe UI", 10),
        )
        style.configure(
            "GuideText.TLabel",
            background="#202331",
            foreground="#E0E3FF",
            font=("Segoe UI", 10),
            wraplength=600,
            justify="left",
        )
        style.configure(
            "GuideNav.TButton",
            anchor="w",
            padding=6,
        )

        main = ttk.Frame(self, style="Guide.TFrame")
        main.pack(fill="both", expand=True, padx=16, pady=16)

        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        
        header_frame = ttk.Frame(main, style="Guide.TFrame")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        title_label = ttk.Label(header_frame,text="User Guide",style="GuideHeader.TLabel",)
        title_label.pack(anchor="w")

        subtitle_label = ttk.Label(header_frame,text=" ",style="GuideSubHeader.TLabel",)
        subtitle_label.pack(anchor="w")

        nav_frame = ttk.Frame(main, style="Guide.TFrame")
        nav_frame.grid(row=1, column=0, sticky="nsw", padx=(0, 16))
        nav_label = ttk.Label(nav_frame,text="Sections",style="GuideSubHeader.TLabel",)
        nav_label.pack(anchor="w", pady=(0, 6))

        self.nav_buttons = {}
        sections = [
            ("getting_started", "Getting Started", True),
            ("workspace", "Workspace Overview", False),
            ("shortcuts", "Keyboard Shortcuts", True),
            ("faq", "FAQ", False),
        ]

        for key, text, access in sections:
            btn = ttk.Button(nav_frame,text=text,style="GuideNav.TButton",command=lambda k=key: self.show_section(k),state=tk.NORMAL if access else tk.DISABLED)
            btn.pack(fill="x", pady=2)
            self.nav_buttons[key] = btn

        content_outer = ttk.Frame(main, style="Guide.TFrame")
        content_outer.grid(row=1, column=1, sticky="nsew")
        content_outer.rowconfigure(0, weight=1)
        content_outer.columnconfigure(0, weight=1)
        
        self.content_canvas = tk.Canvas(content_outer,bg="#202331",highlightthickness=0,borderwidth=0,)
        self.content_canvas.grid(row=0, column=0, sticky="nsew")
        self.content_canvas.configure()
        self.content_canvas.bind("<Configure>", self.on_canvas_configure)
        
        self.content_frame = ttk.Frame(self.content_canvas, style="Guide.TFrame")
        self.content_window = self.content_canvas.create_window((0, 0),window=self.content_frame,anchor="nw",)
        self.content_frame.bind("<Configure>", self.on_content_configure)
        
        self.current_section_key = None
        self.show_section("getting_started")

    def show(self):
        if not self.winfo_exists():
            return
        self.deiconify()
        self.lift()
        self.focus_force()
    def hide(self):
        if self.winfo_exists():
            self.withdraw()
    def on_close(self):
        self.hide()
    def on_content_configure(self, event):
        self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all"))
    def on_canvas_configure(self, event):
        canvas_width = event.width
        self.content_canvas.itemconfig(self.content_window, width=canvas_width)
    def clear_content(self):
        for child in self.content_frame.winfo_children():
            child.destroy()
    def show_section(self, key: str):
        if key == self.current_section_key:
            return
        self.current_section_key = key
        self.clear_content()
        if key == "getting_started":
            self.section_getting_started()
        elif key == "workspace":
            self.section_workspace()
        elif key == "shortcuts":
            self.section_shortcuts()
        else:
            self.section_faq()
    def add_heading(self, text: str):
        lbl = ttk.Label(self.content_frame,text=text,style="GuideHeader.TLabel",)
        lbl.pack(anchor="w", pady=(0, 4))
    def add_subheading(self, text: str):
        lbl = ttk.Label(self.content_frame,text=text,style="GuideSubHeader.TLabel",)
        lbl.pack(anchor="w", pady=(12, 2))
    def add_paragraph(self, text: str):
        lbl = ttk.Label(self.content_frame,text=text,style="GuideText.TLabel",)
        lbl.pack(anchor="w", pady=2)
    def section_getting_started(self):
        self.add_heading("Getting Started")
        self.add_paragraph(
            "This section will give a quick overview of the annotator"
        )

        self.add_subheading("1. Load media")
        self.add_paragraph(
            "* Load a video, folder images\n"
            "* Select the number of frames in a block (the frames handled in one propagation)"
        )

        self.add_subheading("2. Create labels")
        self.add_paragraph(
            "* Create as many labels as you need (each with a unique name)\n"
            "* You can assign permanentfeatures to them"
        )
        self.add_subheading("3. Add promts")
        self.add_paragraph(
            "* You can add prompts, by clicking on the canvas\n"
            "* Positive prompts: this is A\n"
            "* Negative prompt: this is not A\n"
            "* You can add many points and a single box style prompt"
        )
        self.add_subheading("4. Propagation")
        self.add_paragraph(
            "* Once you have added all prompts you can propagate the data across frames\n"
            "* Adding checkpoints will create a block across which the program doesn't propagate\n"
            "  Use this, to mark areas that are good, and any further correction shouldn't affect\n"
            "* If the propagation is successful you will see an overlay view of the generated masks"
        )
        self.add_subheading("5. Export")
        self.add_paragraph(
            "* You can export two things:"
            "* Session data (under export/<Session name>.pkl), so you can continue working from where you left off\n"
            "* And annotation output saved under export/<Session name>/..."
        )
    def section_workspace(self):
        self.add_heading("Workspace Overview")
        self.add_paragraph(
            "Describe the main layout: toolbar, canvas, sidebar panels, status bar, etc."
        )
        self.add_subheading("Main Canvas")
        self.add_paragraph("Explain navigation, zoom, pan, and overlays.")
        self.add_subheading("Panels")
        self.add_paragraph("Describe list of images, layers, properties, etc.")

    def section_shortcuts(self):
        self.add_heading("Keyboard Shortcuts")
        self.add_paragraph(
            "Below is a list of the most important shortcuts to speed up your workflow."
        )
        header_frame = ttk.Frame(self.content_frame, style="Guide.TFrame")
        header_frame.pack(anchor="w", pady=(12, 4), fill="x")
        table = ttk.Frame(self.content_frame, style="Guide.TFrame")
        table.pack(anchor="w", fill="x")
        ttk.Label(table,text="Action",style="GuideSubHeader.TLabel",).grid(row=0, column=0, sticky="w", padx=(0, 20))
        ttk.Label(table,text="Shortcut",style="GuideSubHeader.TLabel",).grid(row=0, column=1, sticky="w")
        
        shortcuts = [
            ("Fullscreen", "Shift + F"),
            ("Next frame", "D"),
            ("Previous frame", "A"),
            ("Next label", "W"),
            ("Previous label", "S"),
            ("Switch to next viewmode", "Shift + M"),
            ("Switch to specific viewmode", "Shift + 1/2/3/4"),
            ("Select prompt (point or box)", "Ctrl + Click"),
            ("Delete selected prompt (point or box)", "Delete"),
        ]
        
        for r, (action, key) in enumerate(shortcuts):
            ttk.Label(table,text=action,style="GuideText.TLabel",).grid(row=r+1, column=0, sticky="w", padx=(0, 20), pady=2)
            ttk.Label(table,text=key,style="GuideText.TLabel",).grid(row=r+1, column=1, sticky="w", pady=2)
            
    def section_faq(self):
        self.add_heading("Frequently Asked Questions")
        self.add_paragraph(
            " - "
        )