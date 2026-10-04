# main.py
import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import NOMBRE_APP, apply_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(NOMBRE_APP)
    # El tema va antes de la ventana para que los widgets nazcan ya con estilo.
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
