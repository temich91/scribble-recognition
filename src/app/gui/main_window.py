import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QSizePolicy, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QPushButton
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap, Qt, QPainter, QPen, QGradient, QColor

MINIMAL_SIZE = QSize(600, 500)

class Painter(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window setup
        self.setMinimumSize(MINIMAL_SIZE)
        self.setWindowTitle("ScribbleRecognizer")

        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)
        self.main_layout.setSpacing(0)

        # Digit drawing side
        paint_widget = QWidget()
        paint_widget.setStyleSheet("background-color: #bcf1ff")

        self.canvas = QWidget(paint_widget)
        self.canvas.setMinimumSize(QSize(300, 300))
        self.canvas.setStyleSheet("background-color: #ffffff")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Buttons for paint actions/options
        self.pen_size_btn = QPushButton("Pen size")
        self.pen_size_btn.setStyleSheet("background-color: orange; color:black")
        self.save_btn = QPushButton("&Save")
        self.save_btn.setStyleSheet("background-color: orange; color:black")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("background-color: orange; color:black")
        paint_options = QHBoxLayout()
        paint_options.addWidget(self.pen_size_btn)
        paint_options.addWidget(self.save_btn)
        paint_options.addWidget(self.clear_btn)

        paint_layout = QVBoxLayout()
        paint_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        paint_layout.addWidget(self.canvas)
        paint_layout.addLayout(paint_options)
        paint_widget.setLayout(paint_layout)

        # Digit probabilities side
        digit_guesses = QWidget()
        digit_guesses.setStyleSheet("background-color: #a2cfdb; color: black; font-size: 18px;")
        digits_layout = QVBoxLayout()
        self.digits_probability = {}
        for i in range(10):
            self.digits_probability[i] = QLabel(f"{i}:")
            digits_layout.addWidget(self.digits_probability[i])
        digit_guesses.setLayout(digits_layout)

        self.main_layout.addWidget(paint_widget, 2)
        self.main_layout.addWidget(digit_guesses, 1)
        self.setCentralWidget(main_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Painter()
    window.show()
    app.exec()
