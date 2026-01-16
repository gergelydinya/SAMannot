from main_window import MainWindow
from ttkthemes import ThemedTk
if __name__ == "__main__":
    root = ThemedTk(theme='yaru')
    app = MainWindow(root)
    root.mainloop()