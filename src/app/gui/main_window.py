import sys
import torch
import torchvision.io
import torchvision.transforms as tfs
from PySide6.QtWidgets import (QMainWindow, QApplication, QWidget, QSizePolicy, QSlider,
                               QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog)
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import Qt
from canvas import Canvas
MINIMAL_SIZE = QSize(600, 500)

class Painter(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window setup
        self.setMinimumSize(MINIMAL_SIZE)
        self.setWindowTitle("ScribbleRecognizer")
        self.statusBar()

        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)
        self.main_layout.setSpacing(0)

        # Digit drawing side
        paint_widget = QWidget()
        paint_widget.setStyleSheet("background-color: #bcf1ff")

        self.canvas = Canvas(parent=paint_widget)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Buttons for paint actions/options

        self.save_btn = QPushButton("&Save")
        self.save_btn.setStyleSheet("background-color: orange; color:black")
        # self.save_btn.clicked.connect(self.save_canvas)
        self.save_btn.clicked.connect(self.getDigitTensor)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("background-color: orange; color:black")
        self.clear_btn.clicked.connect(self.canvas.clear)

        paint_options = QHBoxLayout()
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

    def getDigitTensor(self):
        imgTensor = self.canvas.convertToTensor()
        imgTensor = 255.0 - imgTensor
        _, rows, cols = imgTensor.nonzero(as_tuple=True)
        # Boundaries of the figure in the drawing
        minRow, maxRow = rows.min(), rows.max()
        minCol, maxCol = cols.min(), cols.max()
        croppedTensor = imgTensor[:, minRow: maxRow + 1, minCol: maxCol + 1]

        transform = tfs.Resize((28, 28))
        padding = torch.nn.ConstantPad2d(padding=16, value=0)
        resultTensor = padding(croppedTensor)
        resultTensor = transform(resultTensor).to(dtype=torch.uint8)
        torchvision.io.write_png(resultTensor, "../../../converted.png", 0)
        print("Saved")
        return

    def save_canvas(self):
        filePath, _ = QFileDialog.getSaveFileName(self, caption="caption", dir="../../..")
        filename = filePath.split("/")[-1]
        self.canvas.save(filePath)
        self.statusBar().showMessage(f"Canvas was saved to {filename}.png", timeout=1500)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Painter()
    window.show()
    app.exec()
