import threading
import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk
import time
import numpy as np
from annotator import Annotator
import pickle
from system_monitor import *
from user_guide import *
import pynvml
import json
from tkinter import filedialog
from visual_widget_utils import *
from functional_widget_utils import *
class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("SAMannot")
        self.root.configure(bg="#f5f6f7")
        self.root.tk.call("tk", "scaling", 1.33)
        self.window_width = 1600
        self.window_height = 1080
        self.root.geometry(f"{self.window_width}x{self.window_height}+1+1")
        self.root.update_idletasks()
        self.root.minsize(800, 540)
        self.control_width = 300
        self.current_img_width = 0
        self.current_img_height = 0
        self.keycodes = {}
        
        self.file_path = ""
        self.loading_media = False
        self.image_file_types=["jpg","JPG","jpeg","JPEG","png"]
        self.video_file_types=["mp4","avi","mov","mkv","wmv","flv"]
        self.max_frame = 0
        self.pt_type = 1
        self.currently_selected_feature = -1
        self.label_type = 0
        self.placing_prompt = False
        self.first_point = None
        self.label_markers = []
        self.box_label_markers = []
        
        self.full_screen = False
        self.monitor_window = None
        self.canvas = None
        self.slider_var = tk.IntVar()
        
        self.backend = Annotator()
        self.load_keycodes()
        self.create_gui()
        os.makedirs("data",exist_ok=True)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind_all("<Button-1>", self._global_click, add="+")
        self.root.bind("<Configure>", self.on_window_rsz)
        self.root.bind_all("<KeyPress>", self.process_key_press)
        self.root.bind("<Button-1>", self.clear_focus)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        self.system_monitor = SystemMonitor()
        self.sys_monitor = SystemMonitor()
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self.system_monitor_thread,args=[self.sys_monitor,self.stop_event])
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.root.update_idletasks()
        self.autoload_model()
    def __del__(self):
        self.stop_event.set()
        self.monitor_thread.join()
    # OTHERS    
    def _global_click(self, event):
        if event.widget != self.canvas:
            self.placing_prompt = False
    def system_monitor_thread(self,sys_monitor,stop_event):
        while not stop_event.is_set():
            try:
                sys_monitor.update_system_info()
            except pynvml.NVMLError as e:
                print("NVML error in monitor thread:", e)
            except Exception as e:
                print("Unexpected error in monitor thread:", e)
            self.monitor_window.circular_progress_bar_ram.set_value(int((sys_monitor.used_ram/sys_monitor.total_ram)*100), text=f"{np.round(sys_monitor.used_ram / (1024**3),2)}/{np.round(sys_monitor.total_ram / (1024**3),2)} GB")
            for idx in range(sys_monitor.device_count):
                self.monitor_window.circular_progress_bar_gpu.set_value(int(sys_monitor.gpu_util[idx]))
                self.monitor_window.circular_progress_bar_vram.set_value(int((sys_monitor.gpu_memory[idx][0]/sys_monitor.gpu_memory[idx][1])*100),text=f"{sys_monitor.gpu_memory[idx][0]/(1024**3):.2f}/{sys_monitor.gpu_memory[idx][1]/(1024**3):.2f} GB")
            time.sleep(1.0)
    def clear_focus(self,event):
        if not isinstance(event.widget, (ttk.Entry, tk.Text)):
            self.root.focus_set()
    def hex_to_rgb(self,hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # KEYS
    def load_keycodes(self):
        with open("keycodes_linux.json","r", encoding="utf-8") as f:
            linux = json.load(f)
        with open("keycodes_windows.json","r", encoding="utf-8") as f:
            windows = json.load(f)
        self.keycodes = {"linux": linux, "windows": windows}
    def process_key_press(self, e):
        if isinstance(e.widget, ttk.Entry) or isinstance(e.widget, tk.Text) or isinstance(e.widget, tk.Entry):
            return
        SHIFT_MASK = 0x0001
        CTRL_MASK = 0x0004
        if (os.name == "posix"):
            keycodes = self.keycodes["linux"]
        else:                                    
            keycodes = self.keycodes["windows"]
        if (e.state & SHIFT_MASK) and (not (e.state & CTRL_MASK)):
            if e.keycode == keycodes["1"]:
                self.change_view_mode(0)
            if e.keycode == keycodes["2"]:
                self.change_view_mode(1)
            if e.keycode == keycodes["3"]:
                self.change_view_mode(2)
            if e.keycode == keycodes["4"]:
                self.change_view_mode(3)
            if e.keycode == keycodes["M"]:
                self.change_view_mode(-1)
        if (not (e.state & SHIFT_MASK)) and (e.state & CTRL_MASK):
            if e.keycode == keycodes["F"]:
                if not self.full_screen:
                    self.control_panel.grid_remove()
                    self.root.grid_columnconfigure(0,minsize=0)
                    self.full_screen = True
                else:
                    self.control_panel.grid()
                    self.root.grid_columnconfigure(0,weight=0)
                    self.full_screen = False
                self.update_canvas()
        if (not (e.state & SHIFT_MASK)) and (not (e.state & CTRL_MASK)):
            if e.keycode == keycodes["W"]:
                self.switch_label(-1)
            if e.keycode == keycodes["S"]:
                self.switch_label(1)
            if e.keycode == keycodes["A"]:
                self.switch_img(-1)
            if e.keycode == keycodes["D"]: 
                self.switch_img(1)
            if e.keycode == keycodes["DELETE"]:
                self.delete_selected_point(1)
                self.delete_selected_box(1)
    
    # GUI
    
    def create_gui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        style = ttk.Style(self.root)
        style.configure("Black.TFrame", background="black")
        style.configure("Status.TFrame", background="#e6f2ff")
        style.configure("Status.TLabel", background="#e6f2ff")
        self.control_panel = ttk.Frame(self.root, width=self.control_width)
        self.control_panel.grid(row=0, column=0, sticky='ns', padx=5, pady=5)
        self.control_panel.grid_propagate(False)
        # Session
        session_frame = ttk.LabelFrame(self.control_panel, text="Session")
        session_frame.pack(fill=tk.X, padx=5, pady=5)
        # Session name
        session_name_frame = ttk.Frame(session_frame)
        session_name_frame.pack(fill=tk.X, pady=5)
        self.session_name_label = ttk.Label(session_name_frame,text="Name:")
        self.session_name_label.pack(side=tk.LEFT, pady=5)
        self.session_name_entry = ttk.Entry(session_name_frame)
        self.session_name_entry.pack(fill=tk.X, pady=5)
        self.session_name_entry.bind('<Return>',self.change_session_name)
        # Session controls
        session_control_frame = ttk.Frame(session_frame)
        session_control_frame.pack(fill=tk.X)
        self.load_session_btn = ttk.Button(session_control_frame, text="Load", command=self.load_session)
        self.load_session_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.save_session_btn = ttk.Button(session_control_frame, text="Save", command=self.save_session)
        self.save_session_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.reset_btn = ttk.Button(session_control_frame, text="Reset", command=self.reset_session)
        self.reset_btn.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        session_control_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="buttons")
        export_control_frame = ttk.Frame(session_frame)
        export_control_frame.pack(fill=tk.X)
        self.export_tracking_btn = ttk.Button(export_control_frame, text="Export Annotations", command=self.export_all, state=tk.NORMAL)
        export_control_frame.columnconfigure(0, weight=1)
        self.export_tracking_btn.grid(row=0, column=0, padx=2, pady=(0, 4), sticky="ew")
        # Extra functionalities
        check_box_control_frame = ttk.Frame(session_frame)
        check_box_control_frame.pack(fill=tk.X, padx=2)
        check_box_control_frame.columnconfigure(0, weight=0)
        check_box_control_frame.columnconfigure(1, weight=1)
        check_box_control_frame.columnconfigure(2, weight=0)
        check_box_control_frame.columnconfigure(3, weight=0)
        check_box_control_frame.columnconfigure(4, weight=0)
        check_box_control_frame.columnconfigure(5, weight=1)
        check_box_control_frame.columnconfigure(6, weight=0)
        
        self.open_diagnostics_btn = ttk.Button(check_box_control_frame, text="I",width=1,command=self.open_system_monitor)
        self.open_diagnostics_btn.grid(row=0, column=0, pady=2, sticky="w")
        ToolTip(self.open_diagnostics_btn, "System resources")
        self.cache_check_box_var = tk.BooleanVar(value=False)
        self.cache_check_box = ttk.Checkbutton(check_box_control_frame,text="",variable=self.cache_check_box_var,command=self.switch_img_cache_fn)
        self.cache_check_box.grid(row=0, column=2, sticky="w")
        ToolTip(self.cache_check_box, "Cache images")
        self.show_label_check_box_var = tk.BooleanVar(value=False)
        self.show_label_check_box = ttk.Checkbutton(check_box_control_frame,text="",variable=self.show_label_check_box_var,command=self.update_canvas)
        self.show_label_check_box.grid(row=0, column=3, sticky="w")
        ToolTip(self.show_label_check_box, "Show labels")
        self.auto_prompt_check_box_var = tk.BooleanVar(value=False)
        self.auto_prompt_check_box = ttk.Checkbutton(check_box_control_frame,text="",variable=self.auto_prompt_check_box_var)
        self.auto_prompt_check_box.grid(row=0, column=4, sticky="w")
        ToolTip(self.auto_prompt_check_box, "Auto prompt")
        self.open_user_guide_btn = ttk.Button(check_box_control_frame, text="?",width=1,command=self.open_user_guide)
        self.open_user_guide_btn.grid(row=0, column=6, pady=2, sticky="w")
        ToolTip(self.open_user_guide_btn, "User guide")
        # Media handling
        media_frame = ttk.LabelFrame(self.control_panel, text="Media")
        media_frame.pack(fill=tk.X, padx=5, pady=5)
        self.load_media_btn = ttk.Button(media_frame, text="Load Media", command=self.load_main_folder_unified, width=36)
        self.load_media_btn.pack(fill=tk.X)
        # Labels
        labels_frame = ttk.LabelFrame(self.control_panel, text="Labels")
        labels_frame.pack(fill=tk.X, padx=5, pady=5)
        new_label_frame = ttk.Frame(labels_frame)
        new_label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(new_label_frame, text="New Label:").pack(side=tk.LEFT, padx=2)
        self.new_label_entry = ttk.Entry(new_label_frame, width=15)
        self.new_label_entry.pack(side=tk.LEFT, padx=2)
        self.new_label_entry.bind('<Return>', lambda event: self.add_label())
        self.add_label_btn = ttk.Button(new_label_frame, text="+", command=self.add_label, width=1.1)
        self.add_label_btn.pack(side=tk.LEFT, padx=2)
        selector_row = ttk.Frame(labels_frame)
        selector_row.pack(fill=tk.X, pady=5)
        # Prompts
        prompt_group = ttk.Labelframe(selector_row, text="Prompt type", padding=(6, 4))
        prompt_group.grid(row=0, column=0, padx=(0, 8), sticky="")
        self.label_type_var = tk.IntVar(value=0)
        ttk.Radiobutton(prompt_group, text="Point", variable=self.label_type_var, value=0,command=self.set_label_type).grid(row=0, column=0, padx=4, pady=2, sticky="")
        ttk.Radiobutton(prompt_group, text="Box", variable=self.label_type_var, value=1,command=self.set_label_type).grid(row=0, column=1, padx=4, pady=2, sticky="")
        label_group = ttk.Labelframe(selector_row, text="Label type", padding=(6, 4))
        label_group.grid(row=0, column=1, padx=(8, 0), sticky="")
        self.point_type_var = tk.IntVar(value=1)
        self.point_type_pos = ttk.Radiobutton(label_group, text="Positive", variable=self.point_type_var, value=1,command=self.set_point_type)
        self.point_type_pos.grid(row=0, column=0, padx=4, pady=2, sticky="")
        self.point_type_neg = ttk.Radiobutton(label_group, text="Negative", variable=self.point_type_var, value=0,command=self.set_point_type)
        self.point_type_neg.grid(row=0, column=1, padx=4, pady=2, sticky="")
        selector_row.grid_columnconfigure((0, 1), weight=1)
        # Label listbox
        label_list_frame = ttk.Frame(labels_frame)
        label_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = ttk.Scrollbar(label_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_listbox = EditableListbox(label_list_frame, self.label_edit, self.label_check, height=5, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE,exportselection=0, activestyle='underline', selectforeground='white')
        self.label_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.label_listbox.yview)
        self.label_listbox.bind('<<ListboxSelect>>', self.on_label_select)
        # Features
        new_features_frame = ttk.Frame(labels_frame)
        new_features_frame.pack(fill=tk.X, pady=5)
        ttk.Label(new_features_frame, text="New feature:").pack(side=tk.LEFT, padx=2)
        self.new_feature_entry = ttk.Entry(new_features_frame, width=15)
        self.new_feature_entry.pack(side=tk.LEFT, padx=2)
        self.new_feature_entry.bind('<Return>', lambda event: self.add_feature())
        self.add_feature_btn = ttk.Button(new_features_frame, text="+", command=self.add_feature, width=1.1)
        self.add_feature_btn.pack(side=tk.LEFT, padx=2)
        # Feature list box
        feature_list_frame = ttk.Frame(labels_frame)
        feature_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.feature_label = ttk.Label(feature_list_frame, text="Features",anchor="center",justify="center")
        self.feature_label.pack(fill=tk.X)
        feature_scrollbar = ttk.Scrollbar(feature_list_frame)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_listbox = EditableListbox(feature_list_frame, self.feature_edit, None, height=3, yscrollcommand=feature_scrollbar.set, selectmode=tk.SINGLE,exportselection=0)#tk.Listbox(feature_list_frame, height=3, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE,exportselection=0)
        self.feature_listbox.pack(fill=tk.BOTH, expand=True)
        feature_scrollbar.config(command=self.feature_listbox.yview)
        self.feature_listbox.bind('<<ListboxSelect>>', self.on_feature_select)
        # Prompt lists
        lists_container = ttk.Frame(labels_frame)
        lists_container.pack(fill=tk.BOTH, expand=True, pady=5)
        points_list_frame = ttk.Frame(lists_container)
        points_list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.point_label = ttk.Label(points_list_frame, text="Points", anchor="center", justify="center")
        self.point_label.pack(fill=tk.X)
        scrollbar_points = ttk.Scrollbar(points_list_frame)
        scrollbar_points.pack(side=tk.RIGHT, fill=tk.Y)
        self.points_listbox = tk.Listbox(points_list_frame,height=6,yscrollcommand=scrollbar_points.set,selectmode=tk.EXTENDED)
        self.points_listbox.pack(fill=tk.BOTH, expand=True)
        self.points_listbox.bind("<<ListboxSelect>>", self.on_prompt_select)
        scrollbar_points.config(command=self.points_listbox.yview)
        
        box_list_frame = ttk.Frame(lists_container)
        box_list_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.box_label = ttk.Label(box_list_frame, text="Boxes", anchor="center", justify="center")
        self.box_label.pack(fill=tk.X)
        scrollbar_boxes = ttk.Scrollbar(box_list_frame)
        scrollbar_boxes.pack(side=tk.RIGHT, fill=tk.Y)
        self.box_listbox = tk.Listbox(box_list_frame,height=6,yscrollcommand=scrollbar_boxes.set,selectmode=tk.EXTENDED)
        self.box_listbox.pack(fill=tk.BOTH, expand=True)
        self.box_listbox.bind("<<ListboxSelect>>", self.on_prompt_select)
        scrollbar_boxes.config(command=self.box_listbox.yview)
        # Prompt controls
        lists_container.grid_columnconfigure(0, weight=1)
        lists_container.grid_columnconfigure(1, weight=1)
        lists_container.grid_rowconfigure(0, weight=1)
        points_btn_frame = ttk.Frame(labels_frame, padding=(0, 5))
        points_btn_frame.pack(fill=tk.X)
        self.delete_feature_btn = ttk.Button(points_btn_frame, text="Delete Feature",command=self.delete_selected_feature, state=tk.DISABLED)
        self.delete_feature_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.delete_point_btn = ttk.Button(points_btn_frame, text="Delete Point",command=self.delete_selected_point, state=tk.DISABLED)
        self.delete_point_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.delete_box_btn = ttk.Button(points_btn_frame, text="Delete Box",command=self.delete_selected_box, state=tk.DISABLED)
        self.delete_box_btn.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        points_btn_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="delete")
        # Propagation controls
        video_frame = ttk.LabelFrame(self.control_panel, text="Propagate", labelanchor="n")
        video_frame.pack(fill=tk.X, padx=5, pady=5)
        propagate_frame = ttk.Frame(video_frame)
        propagate_frame.pack(fill=tk.X, pady=5)
        
        self.generate_mask_btn = ttk.Button(propagate_frame, text="Single", command=self.generate_mask, state=tk.DISABLED)
        self.generate_mask_btn.grid(row=0, column=0, pady=2)

        self.init_tracking_btn_backward = ttk.Button(propagate_frame, text="Backward", command=self.run_tracking_backward, state=tk.DISABLED)
        self.init_tracking_btn_backward.grid(row=0, column=1, pady=2)

        self.init_tracking_btn_all = ttk.Button(propagate_frame, text="All", command=self.run_tracking_all, state=tk.DISABLED)
        self.init_tracking_btn_all.grid(row=0, column=2, pady=2)

        self.init_tracking_btn_forward = ttk.Button(propagate_frame, text="Forward", command=self.run_tracking_forward, state=tk.DISABLED)
        self.init_tracking_btn_forward.grid(row=0, column=3, pady=2)

        self.add_remove_block_button = ttk.Button(propagate_frame, text="Add checkpoint", command=self.add_remove_propagation_block, state=tk.DISABLED)
        self.add_remove_block_button.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(5,5))
        propagate_frame.grid_columnconfigure((0,1,2,3), weight=1)
        # View mode
        self.view_mode_frame = ttk.Frame(self.control_panel)
        self.view_mode_frame.pack(fill=tk.BOTH, pady=5)
        self.view_mode_var = tk.IntVar(value=1)
        ttk.Radiobutton(self.view_mode_frame, text="Original", variable=self.view_mode_var, value=0,command=lambda: self.change_view_mode(0)).grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        ttk.Radiobutton(self.view_mode_frame, text="Prompts", variable=self.view_mode_var, value=1,command=lambda: self.change_view_mode(1)).grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        ttk.Radiobutton(self.view_mode_frame, text="Overlay", variable=self.view_mode_var, value=2,command=lambda: self.change_view_mode(2)).grid(row=0, column=2, padx=4, pady=2, sticky="ew")
        ttk.Radiobutton(self.view_mode_frame, text="Masks", variable=self.view_mode_var, value=3,command=lambda: self.change_view_mode(3)).grid(row=0, column=3, padx=4, pady=2, sticky="ew")
        for col in range(4):
            self.view_mode_frame.grid_columnconfigure(col, weight=1)
        # Status frame
        self.status_frame = ttk.Frame(self.control_panel, style="Status.TFrame", padding=(10, 5))
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 5))
        self.status_label = ttk.Label(self.status_frame,text="",style="Status.TLabel",wraplength=self.control_width - 30,anchor="center",justify="center")
        self.status_label.pack(fill=tk.X)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, length=self.control_width-30)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        # Canvas
        self.image_frame = ttk.Frame(self.root, style="Black.TFrame")
        self.image_frame.grid(row=0, column=1, sticky='nsew')
        self.image_frame.grid_rowconfigure(0, weight=0)
        self.image_frame.grid_rowconfigure(1, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.image_frame, bg="black",highlightthickness=0,bd=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.image_counter_label = ttk.Label(self.image_frame, text="",anchor="center",justify="center",background="black", foreground="white",font=("Segoe UI", 11, "bold"))
        self.image_counter_label.grid(row=0, column=0, sticky="n")
        # Navigation
        nav_frame = ttk.Frame(self.image_frame)
        nav_frame.grid(sticky="ew")
        self.slider_canvas = tk.Canvas(nav_frame, highlightthickness=0, height=24)
        self.slider_canvas.grid(row=0, column=2, sticky="ew", padx=4)
        # Timelines
        self.timelines = []
        self.timeline = np.zeros((12,100,3), dtype=np.uint8)
        _ = self.slider_canvas.create_image(0, 0, image=ImageTk.PhotoImage(Image.fromarray(self.timeline)), anchor="nw")
        def resize_timeline_canvas(event):
            global slider_bg_img
            slider_bg_img = ImageTk.PhotoImage(Image.fromarray(self.timeline).resize((event.width, event.height),Image.Resampling.NEAREST))
            slider_bg_id = self.slider_canvas.create_image(0, 0, image=slider_bg_img, anchor="nw")
            self.slider_canvas.itemconfig(slider_bg_id, image=slider_bg_img)
        self.slider_canvas.bind("<Configure>", resize_timeline_canvas)
        # Navigation controls
        self.prev_block_btn = ttk.Button(nav_frame, text="<<", command=lambda: self.switch_block(-1), state=tk.DISABLED)
        self.prev_block_btn.grid(row=1,column=0)
        self.prev_btn = ttk.Button(nav_frame, text="<", command=lambda: self.switch_img(-1), state=tk.DISABLED)
        self.prev_btn.grid(row=1,column=1)
        
        self.next_btn = ttk.Button(nav_frame, text=">", command=lambda: self.switch_img(1), state=tk.DISABLED)
        self.next_btn.grid(row=1,column=3)
        self.next_block_btn = ttk.Button(nav_frame, text=">>", command=lambda: self.switch_block(1), state=tk.DISABLED)
        self.next_block_btn.grid(row=1,column=4)

        self.navigation_slider_var = tk.IntVar()
        self.navigation_slider = ttk.Scale(nav_frame, from_=0, to=0, orient=tk.HORIZONTAL,variable=self.navigation_slider_var)
        self.navigation_slider.grid(row=1,column=2,sticky="ew")
        self.navigation_slider.bind("<ButtonRelease-1>", self.navigation_slider_callback)
        
        nav_frame.grid_columnconfigure(0,weight=0)
        nav_frame.grid_columnconfigure(1,weight=0)
        nav_frame.grid_columnconfigure(2,weight=1)
        nav_frame.grid_columnconfigure(3,weight=0)
        nav_frame.grid_columnconfigure(4,weight=0)
        # Extra windows
        self.monitor_window = SystemMonitorWindow(self.control_panel)
        self.monitor_window.hide()
        self.user_guide_window = UserGuideWindow(self.control_panel)
        self.user_guide_window.hide()
        
    def switch_img_cache_fn(self):
        self.backend.set_cache_images(self.cache_check_box_var.get())
    def change_session_name(self,event):
        self.backend.set_session_name(self.session_name_entry.get().strip())
        self.root.wm_title(self.session_name_entry.get().strip())
        self.session_name_entry.delete(0, tk.END)
    
    # LABEL HANDLING
    
    def add_label(self):
        label_name = self.new_label_entry.get().strip()
        if not label_name:
            self.update_status("Specify the label's name!")
            return
        if self.backend.check_label_existence(label_name):
            self.update_status(f"Label with name '{label_name}' already exists!")
            return
        label_colour = self.backend.add_label(label_name)
        self.label_listbox.insert(tk.END, label_name)
        self.label_listbox.itemconfig(tk.END, {'bg': label_colour}) 
        self.label_listbox.config(selectbackground=label_colour)
        self.label_listbox.selection_clear(0, tk.END)
        self.label_listbox.selection_set(tk.END)
        self.label_listbox.see(tk.END)
        self.label_markers.append([])
        self.box_label_markers.append([])
        self.update_status(f"Added label: {label_name}")
        self.new_label_entry.delete(0, tk.END)
        self.enable_label_controls()
        self.update_pts_list()
        self.update_box_list()
        self.update_features_list()
    
    def update_label_list(self):
        self.label_listbox.delete(0, tk.END)
        for label in self.backend.get_labels():
            self.label_listbox.insert(tk.END, label.name)
            self.label_listbox.itemconfig(tk.END, {'bg': label.col}) 
        self.label_listbox.selection_clear(0, tk.END)
        self.label_listbox.selection_set(tk.END)
        self.label_listbox.see(tk.END)
    def switch_label(self, direction):
        if not self.backend.has_labels():
            return
        label_idx = self.backend.get_current_label_idx()
        if direction == -1:
            new_label_idx = (label_idx - 1)
            if new_label_idx < 0:
                new_label_idx = len(self.backend.get_labels()) - 1
        else:
            new_label_idx = (label_idx + 1) % len(self.backend.get_labels())
        self.label_listbox.selection_clear(0, tk.END)
        self.label_listbox.selection_set(new_label_idx)
        self.label_listbox.activate(new_label_idx)
        self.label_listbox.see(new_label_idx)
        self.label_listbox.event_generate("<<ListboxSelect>>")
    def on_label_select(self, event):
        if not self.backend.has_labels():
            return
        if self.label_listbox.curselection():
            self.backend.set_current_label(self.label_listbox.curselection()[0])
            self.label_listbox.config(selectbackground=self.backend.get_current_label().col)
            self.update_pts_list()
            self.update_box_list()
            self.update_features_list()
            self.timeline[8:10,:,:] = 0
            current_label = self.backend.get_current_label()
            for pts in current_label.pts[self.backend.get_current_block()]:
                self.timeline[8:10,pts.idx, :3] = self.hex_to_rgb(current_label.col)
            for box in current_label.boxes[self.backend.get_current_block()]:
                self.timeline[8:10,box.idx, :3] = self.hex_to_rgb(current_label.col)
            self.timeline[4:6,:,:] = 0
            current_label = self.backend.get_current_label()
            for idx in current_label.prop_frames[self.backend.get_current_block()]:
                self.timeline[4:6,idx, :3] = self.hex_to_rgb(current_label.col)
            self.update_timeline_canvas()
            if self.backend.get_view_mode() == "prompts":
                self.update_canvas()

    def enable_label_controls(self):
        self.delete_point_btn.config(state=tk.NORMAL)
        self.delete_box_btn.config(state=tk.NORMAL)
        self.delete_feature_btn.config(state=tk.NORMAL)
        
    def label_edit(self, label_list):
        self.backend.set_label_name(label_list.edit_index, label_list.entry.get())
        label_idx = label_list.edit_index
        new_label_idx = (label_idx) % len(self.backend.get_labels())
        self.label_listbox.selection_clear(0, tk.END)
        self.label_listbox.selection_set(new_label_idx)
        self.label_listbox.activate(new_label_idx)
        self.label_listbox.see(new_label_idx)
        self.label_listbox.event_generate("<<ListboxSelect>>")
    def label_check(self, label_list):
        for label in self.backend.get_labels():
            if label.name == label_list.entry.get():
                return False
        return True
    # LOAD MEDIA
    
    def load_main_folder_unified(self):
        self.loading_media = True
        self.backend.reset()
        self.root.wm_title(self.backend.get_session_name())
        self.file_path = filedialog.askopenfilename(
            title="Select media",
            filetypes=[
                ("Media files", "*.jpg *.JPG *.jpeg *.JPEG *.png *.mp4 *.avi *.mov *.mkv"),
                ("Image files", "*.jpg *.JPG *.jpeg *.JPEG *.png"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv")
            ],
            initialdir="data"
        )
        if len(self.file_path) == 0:
            self.status_label.config(text=f"Media selection cancelled!")
            return
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Frames to extract")
        progress_dialog.geometry("400x100")
        progress_dialog.transient(self.root)
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        progress_dialog.update_idletasks()
        self.root.update_idletasks()
        progress_dialog.configure(bg="#f5f6f7")
        x = self.root.winfo_rootx() + (self.root.winfo_width() - progress_dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - progress_dialog.winfo_height()) // 2
        progress_dialog.geometry(f"+{x}+{y}")
        
        self.slider_var.set(1)
        self.loading_target_label = ttk.Label(progress_dialog,text = "1")
        self.loading_target_label.pack(fill=tk.X)
        extension = self.file_path.split(".")[-1]
        if extension in self.image_file_types:
            slider = ttk.Scale(progress_dialog, from_=1, to=np.amin([self.backend.get_frame_count_dir(os.path.dirname(self.file_path)),750]), orient=tk.HORIZONTAL,variable=self.slider_var,command=self.update_loading_target)
        else:
            slider = ttk.Scale(progress_dialog, from_=1, to=np.amin([self.backend.get_frame_count(self.file_path),750]), orient=tk.HORIZONTAL,variable=self.slider_var,command=self.update_loading_target)
        slider.pack(fill=tk.X)
        
        self.block_size = 0
        def start_video_loading():
            self.block_size = int(np.round(slider.get()))
            progress_dialog.destroy()
        
        ok_btn = ttk.Button(progress_dialog, text="Ok", command=start_video_loading, width=20, state=tk.NORMAL)
        ok_btn.pack()
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.wait_window(progress_dialog)
        
        self.backend.clear_temp_dir()
        media_type = self.backend.load_main_folder_unified(self.file_path,self.block_size)
        if media_type == 0:
            self.max_frame = self.backend.get_frame_count_dir(os.path.dirname(self.file_path))
        else:
            self.max_frame = self.backend.get_frame_count(self.file_path)
        self.backend.set_num_blocks((self.max_frame // self.block_size + 1) if self.max_frame % self.block_size != 0 else ((self.max_frame // self.block_size)))
        
        self.timelines = []
        for i in range(self.backend.num_blocks):
            timeline = np.zeros((12,self.block_size,3), dtype=np.uint8)
            if i == self.backend.num_blocks - 1:
                if self.max_frame % self.block_size != 0:
                    timeline = np.zeros((12,self.max_frame % self.block_size,3), dtype=np.uint8)
            self.timelines.append(timeline)
        self.timeline = self.timelines[0]
        if self.file_path is not None and len(self.file_path) > 0:
            if media_type == 0:
                self.process_image_folder(os.path.dirname(self.file_path))
            else:
                self.process_video_file(self.file_path)
        self.add_remove_block_button.config(state=tk.NORMAL)
        self.navigation_slider.config(from_=1,to=self.backend.get_loaded_frame_count())
        self.update_label_list()
        self.loading_media = False
        if self.backend.get_current_block() <= 0:
            self.prev_block_btn.config(state=tk.DISABLED)
        else:
            self.prev_block_btn.config(state=tk.NORMAL)
        if self.backend.get_current_block() >= ((self.max_frame // self.block_size) if self.max_frame % self.block_size != 0 else ((self.max_frame // self.block_size) - 1)):
            self.next_block_btn.config(state=tk.DISABLED)
        else:
            self.next_block_btn.config(state=tk.NORMAL)
    def process_image_folder(self, folder_path):
        try:
            if not self.backend.has_frames():
                self.status_label.config(text=f"No image files found in {folder_path}")
                return
            else:
                self.switch_img(1)
                self.status_label.config(text=f"Loaded {np.amin([self.backend.get_number_of_images(),self.block_size])} images from {self.backend.media_path}")
        except Exception as e:
            self.status_label.config(text=f"Error loading folders: {str(e)}")
    def update_loading_target(self,event):
        self.loading_target_label.config(text = str(self.slider_var.get()))
    def process_video_file(self, video_path):
        try:
            self.extract_video_frames(video_path,self.backend.get_current_block() * self.block_size,(self.backend.get_current_block()+1) * self.block_size)
        except Exception as e:
            self.status_label.config(text=f"Error processing video: {str(e)}")
    def extract_video_frames(self, video_path,start_frame,end_frame, interval=1):
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Extracting frames")
        progress_dialog.geometry("400x200")
        progress_dialog.transient(self.root)
        progress_dialog.configure(bg="#f5f6f7")
        ttk.Label(progress_dialog, text="Extracting frames from:").pack(pady=(10, 0))
        ttk.Label(progress_dialog, text=f"{os.path.basename(video_path)}").pack()
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        def extraction_thread():
            try:
                self.backend.extract_frames(start_frame,end_frame,interval,progress_var)
                if self.backend.extracted_frames:
                    self.switch_img(1)
                    self.status_label.config(text=f"Extracted {len(self.backend.extracted_frames)} frames from video")
                progress_dialog.destroy()
            except Exception as e:
                self.status_label.config(text=f"Error extracting frames: {str(e)}")
        extract_thread = threading.Thread(target=extraction_thread)
        extract_thread.daemon = True
        extract_thread.start()
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.wait_window(progress_dialog)
    
    # TRACKING CONTROLS    
    
    def disable_tracking_controls(self):
        self.generate_mask_btn.config(state=tk.DISABLED)
        self.init_tracking_btn_all.config(state=tk.DISABLED)
        self.init_tracking_btn_forward.config(state=tk.DISABLED)
        self.init_tracking_btn_backward.config(state=tk.DISABLED)
    def enable_tracking_controls(self):
        self.generate_mask_btn.config(state=tk.NORMAL)
        self.init_tracking_btn_all.config(state=tk.NORMAL)
        self.init_tracking_btn_forward.config(state=tk.NORMAL)
        self.init_tracking_btn_backward.config(state=tk.NORMAL)
        
    # FEATURE HANDLING
    
    def feature_edit(self, feature_list):
        label = self.backend.get_current_label()
        label.features[feature_list.edit_index].append((feature_list.entry.get(), self.backend.curr_img_idx + self.backend.get_current_block() * self.block_size))
    def add_feature(self):
        feature_name = self.new_feature_entry.get().strip()
        if not feature_name:
            self.update_status("No feature value specified!")
            return
        if self.backend.get_current_label_idx() < 0:
            self.update_status("Please select a label!")
            return
        self.backend.add_feature_to_current_label(feature_name)
        self.feature_listbox.insert(tk.END, feature_name)
        self.feature_listbox.itemconfig(tk.END, {'bg': 'white'}) 
        self.feature_listbox.selection_clear(0, tk.END)
        self.feature_listbox.selection_set(tk.END)
        self.feature_listbox.see(tk.END)
        self.update_status(f"Added feature: {feature_name}")
        self.new_feature_entry.delete(0, tk.END)
    def on_feature_select(self, event):
        if self.feature_listbox.curselection():
            self.currently_selected_feature = self.feature_listbox.curselection()[0]
    def delete_selected_feature(self):
        if self.currently_selected_feature < 0 or self.currently_selected_feature >= len(self.backend.get_current_label().features):
            return
        self.backend.delete_feature_from_current_label(self.currently_selected_feature)
        self.update_features_list()
    def update_features_list(self):
        self.feature_listbox.delete(0, tk.END)
        if not self.backend.get_current_label():
            return
        for i, feature in enumerate(self.backend.get_current_label().features):
            self.feature_listbox.insert(tk.END, self.backend.get_current_label().find_most_recent_feature(i,self.backend.current_block*self.block_size + self.backend.get_current_img_idx()))
    
    # PROMPT HANDLING
    
    def on_prompt_select(self,event):
        if not self.backend.has_labels():
            return
        if self.points_listbox.curselection() and self.backend.get_view_mode() == "prompts":
                self.update_canvas()
    def update_box_list(self):
        self.box_listbox.delete(0, tk.END)
        if self.backend.get_current_label_idx() < 0:
            return
        current_label = self.backend.get_current_label()
        for i, box in enumerate(current_label.boxes[self.backend.get_current_block()]):
            pref = "# " if i == 0 else "x "
            self.box_listbox.insert(tk.END, f"{pref:<2}({box.x}, {box.y}, {box.fx}, {box.fy})-{box.idx}")
    def update_pts_list(self):
        self.points_listbox.delete(0, tk.END)
        if self.backend.get_current_label_idx() < 0:
            return
        current_label = self.backend.get_current_label()
        for i, pt in enumerate(current_label.pts[self.backend.get_current_block()]):
            point_type_str = "(+)" if pt.pt_type == 1 else "(-)"
            self.points_listbox.insert(tk.END, f" ({pt.x}, {pt.y}) - {pt.idx} - {point_type_str}")
    def set_point_type(self):
        self.pt_type = self.point_type_var.get()
    def set_label_type(self):
        self.label_type = self.label_type_var.get()
        if self.label_type == 0:
            self.point_type_pos.config(state=tk.NORMAL)
            self.point_type_neg.config(state=tk.NORMAL)
        if self.label_type == 1:
            self.point_type_var.set(1)
            self.point_type_pos.config(state=tk.DISABLED)
            self.point_type_neg.config(state=tk.DISABLED)
    def refresh_prompt_timeline(self):
        self.timeline[6:8,:,2] = 0
        for label in self.backend.get_labels():
            for pts in label.pts[self.backend.get_current_block()]:
                self.timeline[6:8,pts.idx,2] = 255
            for box in label.boxes[self.backend.get_current_block()]:
                self.timeline[6:8,box.idx,2] = 255
        self.timeline[8:10,:,:] = 0
        current_label = self.backend.get_current_label()
        for pts in current_label.pts[self.backend.get_current_block()]:
            self.timeline[8:10,pts.idx, :3] = self.hex_to_rgb(current_label.col)
        for box in current_label.boxes[self.backend.get_current_block()]:
            self.timeline[8:10,box.idx, :3] = self.hex_to_rgb(current_label.col)
        self.update_timeline_canvas()
    def delete_selected_point(self,flag=0):
        if self.backend.get_current_label_idx() < 0 or self.points_listbox.size() <= 0:
            return
        selection = self.points_listbox.curselection()
        if not selection:
            if flag == 1:
                return
            selection = (self.points_listbox.size()-1),
        cumulative_pt_correction = 0
        for point_index in selection:
            self.backend.delete_selected_point(point_index - cumulative_pt_correction)
            current_label_index = self.backend.get_current_label_idx()
            if point_index < len(self.label_markers[current_label_index]):
                marker_id = self.label_markers[current_label_index][point_index]
                self.label_markers[current_label_index].pop(point_index)
                self.canvas.delete(marker_id)
            cumulative_pt_correction += 1
        self.refresh_prompt_timeline()
        if self.backend.model_status() and self.backend.has_prompts(self.backend.get_current_block()):
            self.enable_tracking_controls()
        else:
            self.disable_tracking_controls()
        self.update_pts_list()
        self.update_box_list()
        self.update_status(f"Deleted points {str(selection)}")
        self.update_canvas()
    def delete_selected_box(self,flag=0):
        if self.backend.get_current_label_idx() < 0 or self.box_listbox.size() <= 0:
            return
        selection = self.box_listbox.curselection()
        if not selection:
            if flag == 1:
                return
            selection = (self.box_listbox.size()-1),
        cumulative_pt_correction = 0
        for box_index in selection:
            self.backend.delete_selected_box(box_index - cumulative_pt_correction)
            current_label_index = self.backend.get_current_label_idx()
            if box_index < len(self.box_label_markers[current_label_index]):
                marker_id = self.box_label_markers[current_label_index][box_index]
                self.box_label_markers[current_label_index].pop(box_index)
                self.canvas.delete(marker_id)
            cumulative_pt_correction += 1
        self.refresh_prompt_timeline()
        if self.backend.model_status() and self.backend.has_prompts(self.backend.get_current_block()):
            self.enable_tracking_controls()
        else:
            self.disable_tracking_controls()
        self.update_box_list()
        self.update_status(f"Deleted boxes {str(selection)}")
        self.update_canvas()
    
    # VISUAL UPDATES    
    
    def redraw_prompts(self):
        if self.backend.get_current_label_idx() < 0:
            return
        selected_label = self.backend.get_current_label()
        selected_label_idx = self.backend.get_current_label_idx()
        selection = self.points_listbox.curselection()
        if not selection:
            selection = (-1,)
        for label_cnt, current_label in enumerate(self.backend.get_labels()):
            self.label_markers[label_cnt] = []
            for pt in current_label.pts[self.backend.get_current_block()]:
                self.label_markers[label_cnt].append(-1)
                if pt.idx != self.backend.get_current_img_idx():
                    continue
                canvas_x = int(pt.x * self.scale_factor) + self.current_img_x + 1
                canvas_y = int(pt.y * self.scale_factor) + self.current_img_y + 1
                marker_radius = 5
                if pt.pt_type == 1:
                    marker_id = self.canvas.create_oval(
                        canvas_x - marker_radius, canvas_y - marker_radius,
                        canvas_x + marker_radius, canvas_y + marker_radius,
                        outline=current_label.col, fill=current_label.col, width=2, tags='marker'
                    )
                else:
                    marker_id = self.canvas.create_oval(
                        canvas_x - marker_radius, canvas_y - marker_radius,
                        canvas_x + marker_radius, canvas_y + marker_radius,
                        outline=current_label.col, fill="black", width=2, tags='marker'
                    )
                
            self.box_label_markers[label_cnt] = []
            for box in current_label.boxes[self.backend.get_current_block()]:
                self.box_label_markers[label_cnt].append(-1)
                if box.idx != self.backend.get_current_img_idx():
                    continue
                f_canvas_x = int(box.fx * self.scale_factor) + self.current_img_x
                f_canvas_y = int(box.fy * self.scale_factor) + self.current_img_y
                canvas_x = int(box.x * self.scale_factor) + self.current_img_x
                canvas_y = int(box.y * self.scale_factor) + self.current_img_y
                marker_id = self.canvas.create_rectangle(
                                f_canvas_x, f_canvas_y,
                                canvas_x, canvas_y,
                                outline=current_label.col, width=5, tags='marker'
                            )  
            if current_label.name == selected_label.name:
                if selection[0] != -1:
                    for pt_idx in selection:
                        pt = current_label.pts[self.backend.get_current_block()][pt_idx]
                        canvas_x = int(pt.x * self.scale_factor) + self.current_img_x + 1
                        canvas_y = int(pt.y * self.scale_factor) + self.current_img_y + 1
                        marker_radius = 5
                        extra_radius = 10
                        marker_id = self.canvas.create_oval(
                            canvas_x - marker_radius - extra_radius, canvas_y - marker_radius - extra_radius,
                            canvas_x + marker_radius + extra_radius, canvas_y + marker_radius + extra_radius,
                            outline="yellow", width=5, tags='marker'
                        )
                        self.label_markers[selected_label_idx].append(marker_id)
                    for box_idx in selection:
                        box = current_label.boxes[self.backend.get_current_block()][box_idx]
                        f_canvas_x = int(box.fx * self.scale_factor) + self.current_img_x
                        f_canvas_y = int(box.fy * self.scale_factor) + self.current_img_y
                        canvas_x = int(box.x * self.scale_factor) + self.current_img_x
                        canvas_y = int(box.y * self.scale_factor) + self.current_img_y
                        marker_id = self.canvas.create_rectangle(
                                        f_canvas_x, f_canvas_y,
                                        canvas_x, canvas_y,
                                        outline="yellow", width=1, tags='marker'
                                    )
                        self.box_label_markers[selected_label_idx].append(marker_id)

    def resize_displayed_img(self):
        img_to_resize = self.backend.get_img_to_resize()
        if img_to_resize is None:
            return
        if not isinstance(img_to_resize,Image.Image):
            img_to_resize = Image.fromarray(img_to_resize)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        orig_w, orig_h = img_to_resize.size
        scale_factor = min(canvas_w / orig_w, canvas_h / orig_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        self.photo_img = ImageTk.PhotoImage(img_to_resize.resize((new_w, new_h), Image.LANCZOS))
        self.canvas.delete("all")
        x_pos = (canvas_w - new_w) // 2
        y_pos = (canvas_h - new_h) // 2
        if x_pos < 0:
            x_pos = 0
        if y_pos < 0:
            y_pos = 0
        self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.photo_img)
        self.current_img_width = new_w
        self.current_img_height = new_h
        self.current_img_x = x_pos
        self.current_img_y = y_pos
        self.scale_factor = scale_factor
        if getattr(self.backend, "view_mode", None) in ("overlay", "masks") and self.show_label_check_box_var.get() == True:
            path = None
            if hasattr(self.backend, "media_files") and hasattr(self.backend, "curr_img_idx"):
                if self.backend.curr_img_idx is not None and self.backend.curr_img_idx >= 0:
                    path = self.backend.media_files[self.backend.curr_img_idx]
            masks_by_path = getattr(self.backend, "masks", None)
            if path is not None and isinstance(masks_by_path, dict) and path in masks_by_path:
                masks_dict = masks_by_path[path]
                font_size = max(8, min(24, int(24 * scale_factor)))
                for label_name, mask in masks_dict.items():
                    if mask is None:
                        continue
                    m = np.asarray(mask)
                    if not np.any(m):
                        continue
                    ys, xs = np.nonzero(m)
                    cy = float(ys.mean())
                    cx = float(xs.mean())
                    x_canvas = int(x_pos + cx * scale_factor)
                    y_canvas = int(y_pos + cy * scale_factor)
                    x_canvas = max(x_pos, min(x_pos + new_w - 1, x_canvas))
                    y_canvas = max(y_pos, min(y_pos + new_h - 1, y_canvas))
                    tmp_id = self.canvas.create_text(x_canvas, y_canvas,text=str(label_name),font=("TkDefaultFont", font_size),anchor=tk.CENTER)
                    x1, y1, x2, y2 = self.canvas.bbox(tmp_id)
                    self.canvas.delete(tmp_id)
                    pad = 3
                    rect_id = self.canvas.create_rectangle(x1 - pad, y1 - pad, x2 + pad, y2 + pad,fill="black",outline="",stipple="gray50")
                    t_shadow = self.canvas.create_text(x_canvas + 1, y_canvas + 1,text=str(label_name),fill="black",font=("TkDefaultFont", font_size),anchor=tk.CENTER)
                    self.canvas.create_text(x_canvas, y_canvas,text=str(label_name),fill="white",font=("TkDefaultFont", font_size),anchor=tk.CENTER)
                    self.canvas.tag_lower(rect_id, t_shadow)
    
    # CANVAS INTERACTION
    
    def check_for_box_overlap(self,click_x,click_y):
        label_cnt = 0
        for current_label in self.backend.get_labels():
            box_cnt = 0
            for box in current_label.boxes[self.backend.get_current_block()]:
                if box.idx != self.backend.get_current_img_idx():
                    continue
                f_canvas_x = int(box.fx * self.scale_factor) + self.current_img_x
                f_canvas_y = int(box.fy * self.scale_factor) + self.current_img_y
                canvas_x = int(box.x * self.scale_factor) + self.current_img_x
                canvas_y = int(box.y * self.scale_factor) + self.current_img_y
                if f_canvas_x <= click_x and click_x <= canvas_x and f_canvas_y <= click_y and click_y <= canvas_y:
                    self.label_listbox.selection_clear(0, "end")
                    self.label_listbox.selection_set(label_cnt)
                    self.label_listbox.activate(label_cnt)
                    self.label_listbox.see(label_cnt)
                    self.label_listbox.event_generate("<<ListboxSelect>>")
                    self.box_listbox.selection_clear(0, "end")
                    self.box_listbox.selection_set(box_cnt)
                    self.box_listbox.activate(box_cnt)
                    self.box_listbox.see(box_cnt)
                    self.box_listbox.event_generate("<<ListboxSelect>>")
                    return
                box_cnt += 1
            label_cnt += 1
    
    def check_for_point_overlap(self,click_x,click_y):
        label_cnt = 0
        marker_radius = 5
        for current_label in self.backend.get_labels():
            point_cnt = 0
            for pt in current_label.pts[self.backend.get_current_block()]:
                if pt.idx != self.backend.get_current_img_idx():
                    continue
                canvas_x = int(pt.x * self.scale_factor) + self.current_img_x
                canvas_y = int(pt.y * self.scale_factor) + self.current_img_y
                if (click_x - canvas_x)**2 + (click_y - canvas_y)**2 <= marker_radius**2:
                    self.label_listbox.selection_clear(0, "end")
                    self.label_listbox.selection_set(label_cnt)
                    self.label_listbox.activate(label_cnt)
                    self.label_listbox.see(label_cnt)
                    self.label_listbox.event_generate("<<ListboxSelect>>")
                    self.points_listbox.selection_clear(0, "end")
                    self.points_listbox.selection_set(point_cnt)
                    self.points_listbox.activate(point_cnt)
                    self.points_listbox.see(point_cnt)
                    self.points_listbox.event_generate("<<ListboxSelect>>")
                    return
                point_cnt += 1
            label_cnt += 1
    def place_point(self, current_label, canvas_x, canvas_y, orig_x, orig_y):
        marker_radius = 5
        fill = "black"
        if self.pt_type == 1:
            fill = current_label.col
        marker_id = self.canvas.create_oval(
            canvas_x - marker_radius, canvas_y - marker_radius,
            canvas_x + marker_radius, canvas_y + marker_radius,
            outline=current_label.col, fill=fill, width=2, tags='marker'
        )
        self.backend.add_point_prompt_to_current_label(orig_x, orig_y, self.pt_type,self.backend.get_current_img_idx())
        self.label_markers[self.backend.get_current_label_idx()].append(marker_id)
        self.update_pts_list()
        self.update_box_list()
        self.refresh_prompt_timeline()
        if self.backend.model_status():
            self.enable_tracking_controls()
        else:
            self.disable_tracking_controls()
        self.update_status(f"Added point at ({orig_x}, {orig_y})")
    def place_box(self,current_label,canvas_x, canvas_y,f_orig_x, f_orig_y, orig_x, orig_y):
        marker_id = self.canvas.create_rectangle(
            self.first_point[0], self.first_point[1], canvas_x, canvas_y,
            outline=current_label.col, width=5, tags='marker'
        )
        if f_orig_x > orig_x:
            f_orig_x, orig_x = orig_x, f_orig_x
        if f_orig_y > orig_y:
            f_orig_y, orig_y = orig_y, f_orig_y
        self.backend.add_box_prompt_to_current_label(f_orig_x, f_orig_y, orig_x, orig_y, self.pt_type,self.backend.get_current_img_idx())
        self.box_label_markers[self.backend.get_current_label_idx()].append(marker_id)
        self.update_pts_list()
        self.update_box_list()
        self.refresh_prompt_timeline()
        self.placing_prompt = False
        self.first_point = None
        if self.backend.model_status():
            self.enable_tracking_controls()
        else:
            self.disable_tracking_controls()
        self.update_status(f"Added box at ({orig_x}, {orig_y})")
    def on_canvas_click(self, event):
        if not self.backend.has_frames() or self.backend.get_current_img_idx() < 0:
            return
        try:
            ctrl_click = ((event.state & 0x0004) > 0)
            if ctrl_click:
                canvas_x, canvas_y = event.x, event.y
                self.check_for_box_overlap(canvas_x,canvas_y)
                self.check_for_point_overlap(canvas_x,canvas_y)
                self.placing_prompt = False
                return
            if self.label_type == 0:
                canvas_x, canvas_y = event.x, event.y
                if (self.current_img_x <= canvas_x <= self.current_img_x + self.current_img_width and
                    self.current_img_y <= canvas_y <= self.current_img_y + self.current_img_height):
                    img_x = canvas_x - self.current_img_x
                    img_y = canvas_y - self.current_img_y
                    orig_x = int(img_x / self.scale_factor)
                    orig_y = int(img_y / self.scale_factor)
                    if self.backend.get_mode() != "prompts" and self.backend.get_mode() != "correction":
                        self.update_status("You cannot add new points while in this mode!")
                        return
                    current_label = self.backend.get_current_label()
                    if not current_label:
                        self.update_status("Please create and select a label first!")
                        return
                    self.place_point(current_label,canvas_x, canvas_y, orig_x, orig_y)
            else:
                canvas_x, canvas_y = event.x, event.y
                if (self.current_img_x <= canvas_x <= self.current_img_x + self.current_img_width and
                    self.current_img_y <= canvas_y <= self.current_img_y + self.current_img_height):
                    if self.placing_prompt == False:
                        if self.backend.get_mode() != "prompts" and self.backend.get_mode() != "correction":
                            self.update_status("You cannot add new points while in this mode!")
                            return
                        current_label = self.backend.get_current_label()
                        if not current_label:
                            self.update_status("Please create and select a label first!")
                            return
                        self.placing_prompt = True
                        self.first_point = (canvas_x,canvas_y)
                    else:
                        img_x = canvas_x - self.current_img_x
                        img_y = canvas_y - self.current_img_y
                        orig_x = int(img_x / self.scale_factor)
                        orig_y = int(img_y / self.scale_factor)
                        f_img_x = self.first_point[0] - self.current_img_x
                        f_img_y = self.first_point[1] - self.current_img_y
                        f_orig_x = int(f_img_x / self.scale_factor)
                        f_orig_y = int(f_img_y / self.scale_factor)
                        if self.backend.get_mode() != "prompts" and self.backend.get_mode() != "correction":
                            self.update_status("You cannot add new points while in this mode!")
                            self.placing_prompt = False
                            self.first_point = None
                            return
                        current_label = self.backend.get_current_label()
                        if not current_label:
                            self.update_status("Please create and select a label first!")
                            self.placing_prompt = False
                            self.first_point = None
                            return
                        self.place_box(current_label,canvas_x, canvas_y,f_orig_x, f_orig_y, orig_x, orig_y)
        except Exception as e:
            self.update_status(f"Error with click event: {str(e)}")
            print(f"Error with click event: {str(e)}")
    
    # AUTO-PROMPTING
    
    def add_multiple_points(self, points, label_name, img_idx,pt_type):
        for (pt_y, pt_x) in points:
            canvas_x = int(pt_x * self.scale_factor + self.current_img_x)
            canvas_y = int(pt_y * self.scale_factor + self.current_img_y)
            orig_x = pt_x
            orig_y = pt_y
            labels = self.backend.get_labels()
            processed_label = labels[self.backend.get_label_idx(label_name)]
            marker_radius = 5
            if pt_type == 1:
                marker_id = self.canvas.create_oval(
                    canvas_x - marker_radius, canvas_y - marker_radius,
                    canvas_x + marker_radius, canvas_y + marker_radius,
                    outline=processed_label.col, fill=processed_label.col, width=2, tags='marker'
                )
            else:
                marker_id = self.canvas.create_oval(
                    canvas_x - marker_radius, canvas_y - marker_radius,
                    canvas_x + marker_radius, canvas_y + marker_radius,
                    outline=processed_label.col, fill="black", width=2, tags='marker'
                )
            self.backend.add_point_prompt_to_label(orig_x, orig_y, pt_type, img_idx, self.backend.get_label_idx(label_name))
            self.label_markers[self.backend.get_label_idx(label_name)].append(marker_id)
            self.update_pts_list()
            self.update_box_list()
            self.timeline[6:8,img_idx ,2] = 255
            self.update_timeline_canvas()
    
    # TRACKING
            
    def add_remove_propagation_block(self):
        img_idx = self.backend.get_current_img_idx()
        if self.backend.has_propagation_block(img_idx):
            self.backend.remove_propagation_block(img_idx)
            self.timeline[10:,img_idx,0] = 0
        else:
            self.backend.add_propagation_block(img_idx)
            self.timeline[10:,img_idx,0] = 255
        self.update_timeline_canvas()
        self.refresh_propagation_block_button()
    def run_tracking_all(self):
        return self.run_tracking(0)
    def run_tracking_forward(self):
        return self.run_tracking(1)
    def run_tracking_backward(self):
        return self.run_tracking(2)
    def run_tracking(self,flag=0):
        if not self.backend.model_status():
            self.update_status("Could not initialize the SAM2 model.")
            return
        if self.backend.get_number_of_images() == 0:
            self.update_status("No frames are loaded.")
            return
        if self.backend.get_mode() == "tracking":
            self.update_status("Propagation is already running.")
            return
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("SAM2 Mask Propagation")
        progress_dialog.geometry("400x100")
        progress_dialog.transient(self.root)
        progress_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        progress_dialog.update_idletasks()
        progress_dialog.configure(bg="#f5f6f7")
        x = self.root.winfo_rootx() + (self.root.winfo_width() - progress_dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - progress_dialog.winfo_height()) // 2
        progress_dialog.geometry(f"+{x}+{y}")
        ttk.Label(progress_dialog, text="Propagating to frames").pack(pady=(10,0))
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        status_label = ttk.Label(progress_dialog, text="Initializing")
        status_label.pack(pady=5)
        def update_progress(message, progress=None):
            status_label.config(text=message)
            if progress is not None:
                progress_var.set(progress)
            progress_dialog.update_idletasks()
        def tracking_thread():
            try:
                update_progress("Initializing tracking state...", 50)
                success = self.backend.initialize_tracking(lambda msg: self.root.after(0, lambda: update_progress(msg)))
                if not success:
                    self.root.after(0, lambda: self.update_status("Failed to initialize tracking."))
                    self.root.after(2000, progress_dialog.destroy)
                    return
                update_progress("Tracking initialized successfully. Starting mask propagation...", 0)
                success, prop_frames = self.backend.propagate(flag,(lambda msg, prog: self.root.after(0, lambda: update_progress(msg, prog))))
                if not success:
                    return
                self.root.after(0, lambda: self.update_status("Masks propagated successfully."))
                update_progress("Propagation complete. Applying masks to all images...", 0)
                self.backend.apply_masks(update_progress)
                self.root.after(0, lambda: self.update_canvas())
                self.root.after(0, lambda: self.update_status("Masks applied to all images."))
                self.root.after(0, lambda: self.export_tracking_btn.config(state=tk.NORMAL))
                #self.root.after(0, self.toggle_view_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.update_view_mode_button())
                self.root.after(500, progress_dialog.destroy)
                self.timeline[4:6,:,:2] = 0
                current_label = self.backend.get_current_label()
                for idx in current_label.prop_frames[self.backend.get_current_block()]:
                    self.timeline[4:6,idx, :3] = self.hex_to_rgb(current_label.col)
                for idx in prop_frames:
                    self.timeline[2:4,idx,1] = 255
                self.update_timeline_canvas()
            except Exception as e:
                self.backend.reset_view_mode()
                self.update_view_mode_button()
                update_progress(f"Error during propagation: {str(e)}", 0)
                self.root.after(0, lambda: self.update_status(f"Error during propagation: {str(e)}"))
                self.root.after(500, progress_dialog.destroy)
        threading.Thread(target=tracking_thread, daemon=True).start()
    def generate_mask(self):
        if not self.backend.model_status():
            self.update_status("Could not initialize the SAM2 model.")
            return
        if self.backend.get_number_of_images() == 0:
            self.update_status("No frames are loaded.")
            return
        if self.backend.get_current_label_idx() < 0 or not self.backend.has_frames():
            return
        self.update_status(f"Generating mask")
        threading.Thread(target=self._generate_mask_thread).start()
    def _generate_mask_thread(self):
        def update_progress(message, progress=None):
            pass
        success, prop_frames = self.backend.generate_mask()
        if not success:
            return
        self.root.after(0, lambda: self.export_tracking_btn.config(state=tk.NORMAL))
        self.backend.apply_masks(update_progress)
        self.root.after(0, lambda: self.update_canvas())
    
    # VIEW MODE
    
    def update_view_mode_button(self):
        if self.backend.get_view_mode() == "original":
            self.view_mode_var.set(0)
        elif self.backend.get_view_mode() == "prompts":
            self.view_mode_var.set(1)
        elif self.backend.get_view_mode() == "overlay":
            self.view_mode_var.set(2)
        else:
            self.view_mode_var.set(3)
            
    def change_view_mode(self, vm_idx = -1):
        if vm_idx == -1:
            self.backend.toggle_view_mode()
        else:
            self.backend.set_view_mode(vm_idx)
        if self.backend.get_view_mode() == "original":
            self.view_mode_var.set(0)
            self.update_canvas()
            self.update_status("Viewing original image")
        elif self.backend.get_view_mode() == "prompts":
            self.view_mode_var.set(1)    
            self.update_canvas()
            self.update_status("Viewing original image with points")
        elif self.backend.get_view_mode() == "overlay":
            self.view_mode_var.set(2) 
            self.update_canvas()
            self.update_status("Viewing overlayed image")
        else:
            self.view_mode_var.set(3)
            self.update_canvas()
            self.update_status("Viewing segmentation masks")
        self.update_view_mode_button()
    
    # LOAD SAM2
    
    def autoload_model(self):
        if self.backend.can_initialize_model(): 
            self.update_status("SAM2 model file found. Loading in background")
            thread = threading.Thread(target=self._load_model_thread)
            thread.daemon = True
            thread.start()
    def _load_model_thread(self):
        success = self.backend.load_model(status_callback=self.update_loading)
        if not success:
            self.root.after(0, lambda: self.update_status("Failed to load SAM2 model"))
    def update_loading(self, message, is_final=False):
        if threading.current_thread() is threading.main_thread():
            if not self.status_frame.winfo_ismapped():
                self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, before=self.status_frame, pady=(0, 5))
            self.status_label.config(text=message)
            if "[" in message and "]" in message and "Loading SAM2 model" in message:
                progress_text = message.split("[")[1].split("]")[0]
                current, total = map(int, progress_text.split("/"))
                progress = (current / total) * 100
                self.progress_var.set(progress)
            if is_final:
                self.root.after(500, self.status_frame.pack_forget())
            self.root.update_idletasks()
        else:
            self.root.after(0, lambda: self.update_loading(message, is_final))
    def update_status(self, message):
        if threading.current_thread() is threading.main_thread():
            self.status_label.config(text=message)
            self.root.update_idletasks()
        else:
            self.root.after(0, lambda: self.status_label.config(text=message))
    
    # DATA EXPORT
    
    def update_export_progress(self, message, status_label, progress_var, progress_dialog, progress=None):
        status_label.config(text=message)
        if progress is not None:
            progress_var.set(progress)
        progress_dialog.update_idletasks()
    def export_all(self):
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Export")
        progress_dialog.geometry("400x100")
        progress_dialog.transient(self.root)
        progress_dialog.protocol("WM_DELETE_WINDOW",lambda: None)
        try:
            progress_dialog.update_idletasks()
            progress_dialog.configure(bg="#f5f6f7")
            x = self.root.winfo_rootx() + (self.root.winfo_width() - progress_dialog.winfo_width()) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - progress_dialog.winfo_height()) // 2
            progress_dialog.geometry(f"+{x}+{y}")
            ttk.Label(progress_dialog,text="Exporting...").pack(pady=(10,0))
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var,maximum=100)
            progress_bar.pack(fill=tk.X,padx=20,pady=10)
            status_label = ttk.Label(progress_dialog, text="Exporting overlays...")
            status_label.pack(pady=5)
            self.backend.init_export()
            def update_export_progress_backend(mode, current_step,total_steps):
                if mode == "overlay":
                    self.update_export_progress(f"Exporting overlays: {current_step+1}/{total_steps}",status_label,progress_var,progress_dialog,(current_step/total_steps)*100)
                elif mode == "mask":
                    self.update_export_progress(f"Exporting masks: {current_step+1}/{total_steps}",status_label,progress_var,progress_dialog,(current_step/total_steps)*100)
                elif mode == "interpolated":
                    self.update_export_progress(f"Exporting interpolated data: {current_step+1}/{total_steps}",status_label,progress_var,progress_dialog,(current_step/total_steps)*100)
                else:
                    pass
            self.backend.export_overlays(update_export_progress_backend)
            self.backend.export_masks(update_export_progress_backend)
            self.backend.export_interpolated_data(update_export_progress_backend)
            self.update_export_progress(f"Exporting manual data...",status_label,progress_var,progress_dialog,0)
            self.backend.export_manual_data()
            self.update_export_progress(f"Exporting label data...",status_label,progress_var,progress_dialog,0)
            self.backend.export_label_data()
        except Exception as e:
            print("Unexpected error during export:", e)
        self.root.after(500, progress_dialog.destroy)

    # NAVIGATION
    
    def navigation_slider_callback(self,event):
        if not self.backend.has_frames():
            return
        self.timeline[:2,self.backend.get_current_img_idx(),:2] = 0
        self.backend.set_img(int(np.round(self.navigation_slider.get()))-1)
        self.timeline[:2,self.backend.get_current_img_idx(),:2] = 255
        self.update_timeline_canvas()
        self.update_canvas()
        self.update_view_mode_button()
        self.update_img_info()
        self.refresh_propagation_block_button()
        self.next_btn.config(state=tk.NORMAL)
        if self.backend.get_current_img_idx() >= self.backend.get_number_of_images() - 1:
            self.next_btn.config(state=tk.DISABLED)
        else:
            self.next_btn.config(state=tk.NORMAL)
            
        if self.backend.get_current_img_idx() <= 0:
            self.prev_btn.config(state=tk.DISABLED)
        else:
            self.prev_btn.config(state=tk.NORMAL)
    def refresh_propagation_block_button(self):
        if self.backend.has_propagation_block(self.backend.get_current_img_idx()):
            self.add_remove_block_button.config(text = "Remove checkpoint")
        else:
            self.add_remove_block_button.config(text = "Add checkpoint")
            
    def switch_img(self,direction=1):
        if not self.backend.has_frames():
            return
        if direction == 1:
            self.backend.next_img()
        else:
            self.backend.prev_img()
        self.update_canvas()
        self.update_view_mode_button()
        self.update_img_info()
        self.update_features_list()
        self.refresh_propagation_block_button()
        self.navigation_slider_var.set(self.backend.get_current_img_idx() + 1)
        self.timeline[:2,:,:2] = 0
        self.timeline[:2,self.backend.get_current_img_idx(),:2] = 255
        self.update_timeline_canvas()
        if self.backend.get_current_img_idx() >= self.backend.get_number_of_images() - 1:
            self.next_btn.config(state=tk.DISABLED)
        else:
            self.next_btn.config(state=tk.NORMAL)
            
        if self.backend.get_current_img_idx() <= 0:
            self.prev_btn.config(state=tk.DISABLED)
        else:
            self.prev_btn.config(state=tk.NORMAL)
    def switch_block(self,direction=1):
        self.next_block_btn.config(state=tk.DISABLED)
        self.prev_block_btn.config(state=tk.DISABLED)
        if self.file_path == "":
            return
        self.backend.reset_media()
        self.backend.set_current_block(self.backend.get_current_block() + direction)
        self.timeline = self.timelines[self.backend.get_current_block()]
        self.timeline[:2,:,:2] = 0
        self.update_timeline_canvas()
        media_type = self.backend.load_main_folder_unified(self.file_path,self.block_size)
        if self.file_path is not None and len(self.file_path) > 0:
            if media_type == 0:
                self.process_image_folder(os.path.dirname(self.file_path))
            elif media_type == 1:
                self.process_video_file(self.file_path)
            else:
                pass
        self.add_remove_block_button.config(state=tk.NORMAL)
        self.navigation_slider.config(from_=1,to=self.backend.get_loaded_frame_count())
        extension = self.file_path.split(".")[-1]
        if extension in self.image_file_types:
            self.max_frame = self.backend.get_frame_count_dir(os.path.dirname(self.file_path))
        else:
            self.max_frame = self.backend.get_frame_count(self.file_path)
        if direction == 1 and (self.backend.get_current_block()-1) in self.backend.extra_frame_masks:
            if self.auto_prompt_check_box_var.get():
                for l, m in self.backend.extra_frame_masks[self.backend.get_current_block()-1].items():
                    self.add_multiple_points(self.backend.generate_point_prompts(m),l,0,1)
        if self.backend.get_current_block() <= 0:
            self.prev_block_btn.config(state=tk.DISABLED)
        else:
            self.prev_block_btn.config(state=tk.NORMAL)
        if self.backend.model_status() and self.backend.has_prompts(self.backend.get_current_block()):
            self.enable_tracking_controls()
        else:
            self.disable_tracking_controls()
        if self.backend.get_current_block() >= ((self.max_frame // self.block_size) if self.max_frame % self.block_size != 0 else ((self.max_frame // self.block_size) - 1)):
            self.next_block_btn.config(state=tk.DISABLED)
        else:
            self.next_block_btn.config(state=tk.NORMAL)
    
    # SESSION EXPORT
    
    def load_session_from_dict(self,frontend_representation):
        self.pt_type = frontend_representation["pt_type"]
        self.currently_selected_feature = frontend_representation["currently_selected_feature"]
        self.label_markers = frontend_representation["label_markers"]
        self.box_label_markers = frontend_representation["box_label_markers"]
        self.timelines = frontend_representation["timelines"]
        self.timeline = frontend_representation["timeline"]
        self.delete_feature_btn["state"] = frontend_representation["delete_feature_btn"]
        self.delete_point_btn["state"] = frontend_representation["delete_point_btn"]
        self.delete_box_btn["state"] = frontend_representation["delete_box_btn"]
        self.generate_mask_btn["state"] = frontend_representation["generate_mask_btn"]
        self.add_remove_block_button["state"] = frontend_representation["add_remove_block_button"]
        self.init_tracking_btn_all["state"] = frontend_representation["init_tracking_btn_all"]
        self.init_tracking_btn_forward["state"] = frontend_representation["init_tracking_btn_forward"]
        self.init_tracking_btn_backward["state"] = frontend_representation["init_tracking_btn_backward"]
        self.export_tracking_btn["state"] = frontend_representation["export_tracking_btn"]
        self.prev_btn["state"] = frontend_representation["prev_btn"]
        self.next_btn["state"] = frontend_representation["next_btn"]
        self.prev_block_btn["state"] = frontend_representation["prev_block_btn"]
        self.next_block_btn["state"] = frontend_representation["next_block_btn"]
        self.image_counter_label["text"] = frontend_representation["image_counter_label"]
        self.status_label["text"] = frontend_representation["status_label"]
        self.max_frame = frontend_representation["max_frame"]
        self.file_path = frontend_representation["file_path"]
    
    def load_session(self):
        session_file_path = filedialog.askopenfilename(
            title="Select media",
            filetypes=[
                ("Cache files", "*.pkl"),
                ("All files", "*.*")
            ],
            initialdir="export"
        )
        if session_file_path is not None and len(session_file_path) > 0:
            with open(session_file_path,"rb") as dtf:
                backend_representation = pickle.load(dtf)
                frontend_representation = pickle.load(dtf)
                self.backend.load_from_dict(backend_representation)
                self.load_session_from_dict(frontend_representation)
            temp_img_idx = self.backend.get_current_img_idx()
            self.block_size = self.backend.block_size
            self.backend.clear_temp_dir()
            if self.backend.video_name == "":
                media_type = self.backend.load_main_folder_unified(self.backend.media_path,self.block_size)
            else:
                media_type = self.backend.load_main_folder_unified(self.backend.video_name,self.block_size)
            self.backend.set_num_blocks((self.max_frame // self.block_size + 1) if self.max_frame % self.block_size != 0 else ((self.max_frame // self.block_size)))
            if media_type == 0:
                self.process_image_folder(os.path.dirname(self.backend.media_path))
            else:
                self.process_video_file(self.backend.media_path)
            self.backend.set_img(temp_img_idx)
            self.navigation_slider.config(from_=1,to=self.backend.get_loaded_frame_count())
            self.navigation_slider_var.set(self.backend.get_current_img_idx() + 1)
            self.root.wm_title(self.backend.get_session_name())
            self.update_pts_list()
            self.update_box_list()
            self.update_features_list()
            self.update_label_list()
            self.update_canvas()
            self.label_listbox.config(selectbackground=self.backend.get_current_label().col)
            if self.backend.get_current_block() <= 0:
                self.prev_block_btn.config(state=tk.DISABLED)
            else:
                self.prev_block_btn.config(state=tk.NORMAL)
            if self.backend.get_current_block() >= ((self.max_frame // self.block_size) if self.max_frame % self.block_size != 0 else ((self.max_frame // self.block_size) - 1)):
                self.next_block_btn.config(state=tk.DISABLED)
            else:
                self.next_block_btn.config(state=tk.NORMAL)
    
    def save_session(self):
        frontend_representation = {
            "pt_type": self.pt_type,
            "currently_selected_feature": self.currently_selected_feature,
            "delete_feature_btn": str(self.delete_feature_btn["state"]),
            "delete_point_btn": str(self.delete_point_btn["state"]),
            "delete_box_btn": str(self.delete_box_btn["state"]),
            "generate_mask_btn": str(self.generate_mask_btn["state"]),
            "add_remove_block_button": str(self.add_remove_block_button["state"]),
            "init_tracking_btn_all": str(self.init_tracking_btn_all["state"]),
            "init_tracking_btn_forward": str(self.init_tracking_btn_forward["state"]),
            "init_tracking_btn_backward": str(self.init_tracking_btn_backward["state"]),
            "export_tracking_btn": str(self.export_tracking_btn["state"]),
            "prev_btn": str(self.prev_btn["state"]),
            "next_btn": str(self.next_btn["state"]),
            "prev_block_btn": str(self.prev_block_btn["state"]),
            "next_block_btn": str(self.next_block_btn["state"]),
            "image_counter_label": str(self.image_counter_label["text"]),
            "status_label": str(self.status_label["text"]),
            "label_markers": self.label_markers,
            "box_label_markers": self.box_label_markers,
            "timelines": self.timelines,
            "timeline": self.timeline,
            "max_frame": self.max_frame,
            "file_path": self.file_path
        }
        os.makedirs("./export", exist_ok=True)
        path = f"./export/{self.backend.session_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.backend.compress_to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(frontend_representation, f, protocol=pickle.HIGHEST_PROTOCOL)

    def reset_session(self):
        del self.backend
        self.backend = Annotator()
        self.pt_type = 1
        self.currently_selected_feature = -1
        self.label_markers = []
        self.box_label_markers = []
        self.max_frame = 0
        self.update_pts_list()
        self.update_box_list()
        self.update_features_list()
        self.update_label_list()
        self.new_label_entry.delete(0, tk.END)
        self.enable_label_controls()
        self.disable_tracking_controls()
        self.add_remove_block_button.config(state=tk.DISABLED)
        self.timelines = []
        self.timeline = np.zeros((12,100,3), dtype=np.uint8)
        self.update_timeline_canvas()
        self.update_img_info()
        self.root.wm_title("SAMannot")
        self.canvas.delete("all")
        self.autoload_model()
    
    # CANVAS UPDATES
    
    def update_canvas(self):
        if self.canvas is None:
            return
        self.canvas.update() 
        if self.backend.get_view_mode() == "original":
            self.root.after(100, self.resize_displayed_img)
        elif self.backend.get_view_mode() == "prompts":
            self.root.after(100, self.resize_displayed_img)
            self.root.after(100, self.redraw_prompts)
        elif self.backend.get_view_mode() == "overlay":
            self.root.after(100, self.resize_displayed_img)
        else:
            self.root.after(100, self.resize_displayed_img)
    def update_img_info(self):
        if not self.backend.has_frames():
            self.image_counter_label.config(text=f"")
            return
        total_images = self.backend.get_number_of_images()
        current_num = self.backend.get_current_img_idx() + 1
        self.image_counter_label.config(text=f"Image: {current_num} / {total_images} | Block: {self.backend.get_current_block() + 1} / {((self.max_frame // self.block_size) if self.max_frame % self.block_size != 0 else ((self.max_frame // self.block_size) - 1)) + 1}")
    def update_timeline_canvas(self):
        global slider_bg_img
        slider_bg_img = ImageTk.PhotoImage(Image.fromarray(self.timeline).resize((self.slider_canvas.winfo_width(), self.slider_canvas.winfo_height()),Image.Resampling.NEAREST))
        slider_bg_id = self.slider_canvas.create_image(0, 0, image=slider_bg_img, anchor="nw")
        self.slider_canvas.itemconfig(slider_bg_id, image=slider_bg_img)
            
    # WINDOW EVENT HANDLING
    
    def on_window_rsz(self, event):
        if event.widget == self.root:
            self.update_canvas()
    def on_close(self):
        self.root.destroy()
    def open_system_monitor(self):
        self.monitor_window.show()
    def open_user_guide(self):
        self.user_guide_window.show()