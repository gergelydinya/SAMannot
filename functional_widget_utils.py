import threading
import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk
import time
import numpy as np
from annotator import Annotator
import pickle
from system_monitor import SystemMonitor
from user_guide import UserGuideWindow
import pynvml
import json
class EditableListbox(tk.Listbox):
    def __init__(self, master, edit_fn, check_edit, **kwargs):
        super().__init__(master, **kwargs)
        self.entry = None
        self.edit_index = None
        self.bind("<Double-1>", self.start_edit)
        self.bind("<MouseWheel>", self.on_scroll)
        self.bind("<Button-4>", self.on_scroll)
        self.bind("<Button-5>", self.on_scroll)
        self.edit_fn = edit_fn
        self.check_edit = check_edit
    def start_edit(self, event):
        index = self.curselection()
        if not index:
            return
        self.edit_index = index[0]
        _, y, _, h = self.bbox(self.edit_index)
        full_width = self.winfo_width()
        if self.entry:
            self.entry.destroy()
        self.entry = tk.Entry(self.master)
        self.entry.insert(0, self.get(self.edit_index))
        self.entry.place(x=self.winfo_x(), y=self.winfo_y()+y, width=full_width, height=h)

        self.entry.focus()
        self.entry.bind("<Return>", self.save_edit)
        self.entry.bind("<FocusOut>", self.save_edit)
    def save_edit(self, event=None):
        if self.entry and self.edit_index is not None:
            editable = True
            if self.check_edit is not None:
                editable = self.check_edit(self)
            if editable:
                item_bg = self.itemcget(self.edit_index, "bg")
                item_fg = self.itemcget(self.edit_index, "fg")
                self.delete(self.edit_index)
                self.insert(self.edit_index, self.entry.get())
                self.itemconfig(self.edit_index, bg=item_bg, fg=item_fg)
                if self.edit_fn is not None:
                    self.edit_fn(self)
        self.cancel_edit()

    def cancel_edit(self):
        if self.entry:
            self.entry.destroy()
        self.entry = None
        self.edit_index = None

    def on_scroll(self, event):
        self.update_idletasks()
        if self.entry and self.edit_index is not None:
            bbox = self.bbox(self.edit_index)
            if bbox:
                _, y, _, h = bbox
                full_width = self.winfo_width()
                self.entry.place(x=self.winfo_x(), y=self.winfo_y()+y, width=full_width, height=h)
            else:
                self.cancel_edit()