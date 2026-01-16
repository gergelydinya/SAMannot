import tkinter as tk
class CircularProgress(tk.Canvas):
    def __init__(self, parent, size=200, thickness=15,bg_color="#f5f5f5", track_color="#d0d4dc",progress_color_low="#6e5bff", progress_color_high="#ff4b7d", text_color="#222222",font=("Segoe UI", 11, "bold"), *args, **kwargs):
        super().__init__(parent, width=size, height=size,bg=bg_color, highlightthickness=0, *args, **kwargs)
        self.colors = self.color_gradient(progress_color_low,progress_color_high,  101)
        pad = thickness // 2 + 4
        self.radius_box = (pad, pad, size - pad, size - pad)
        self.bg_arc_id = self.create_arc(self.radius_box,start=0, extent=359.9,outline=track_color,width=thickness,style=tk.ARC)
        self.arc_id = self.create_arc(self.radius_box,start=90, extent=0,outline=progress_color_low,width=thickness,style=tk.ARC)
        self.text_id = self.create_text(size//2, size//2,text="0%",fill=text_color,font=font)
    def hex_to_rgb(self,hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    def rgb_to_hex(self,rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)
    def color_gradient(self,start_hex, end_hex, steps):
        start_rgb = self.hex_to_rgb(start_hex)
        end_rgb = self.hex_to_rgb(end_hex)
        gradient = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 1
            rgb = tuple(int(start_rgb[j] + (end_rgb[j] - start_rgb[j]) * t) for j in range(3))
            gradient.append(self.rgb_to_hex(rgb))
        return gradient
    def set_value(self, value: float, text = None):
        value = max(0, min(100, float(value)))
        angle = (value / 100) * 360
        self.itemconfig(self.arc_id, extent=-angle)
        self.itemconfig(self.arc_id, outline=self.colors[int(value)])
        if text is None:
            self.itemconfig(self.text_id, text=str(int(value))+"%")
        else:
            self.itemconfig(self.text_id, text=text)
            
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        label = tk.Label(self.tip_window,text=self.text,background="#ffffff",relief="solid",borderwidth=1,padx=5,pady=3)
        label.pack()
        self.tip_window.update_idletasks()
        widget_x = self.widget.winfo_rootx()
        widget_y = self.widget.winfo_rooty()
        widget_w = self.widget.winfo_width()
        tip_w = self.tip_window.winfo_width()
        tip_h = self.tip_window.winfo_height()
        x = widget_x + (widget_w // 2) - (tip_w // 2)
        y = widget_y - tip_h - 5
        self.tip_window.wm_geometry(f"{tip_w}x{tip_h}+{x}+{y}")
    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None