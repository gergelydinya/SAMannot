from ttkthemes import ThemedTk
from .main_window import MainWindow


def main():
    root = ThemedTk(theme='yaru')
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()