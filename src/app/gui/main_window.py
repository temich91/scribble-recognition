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

class MainWindow(QMainWindow):
    """Main window class.
    """
    def __init__(self):
        """ Initialize window properties and layout.

        """
        super().__init__()
        # Window setup
        self.setMinimumSize(MINIMAL_SIZE)
        self.setWindowTitle("ScribbleRecognizer")
        self.statusBar()

        mainWidget = QWidget()
        self.mainLayout = QHBoxLayout(mainWidget)
        self.mainLayout.setSpacing(0)

        # Digit drawing side
        paintWidget = QWidget()
        paintWidget.setStyleSheet("background-color: #bcf1ff")

        self.canvas = Canvas(parent=paintWidget)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Buttons for paint actions/options
        self.saveBtn = QPushButton("&Save")
        self.saveBtn.setStyleSheet("background-color: orange; color:black")
        # self.save_btn.clicked.connect(self.save_canvas)
        self.saveBtn.clicked.connect(self.getDigitTensor)

        self.clearBtn = QPushButton("Clear")
        self.clearBtn.setStyleSheet("background-color: orange; color:black")
        self.clearBtn.clicked.connect(self.canvas.clear)

        # Layouts
        paintOptions = QHBoxLayout()
        paintOptions.addWidget(self.saveBtn)
        paintOptions.addWidget(self.clearBtn)

        paintLayout = QVBoxLayout()
        paintLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        paintLayout.addWidget(self.canvas)
        paintLayout.addLayout(paintOptions)

        paintWidget.setLayout(paintLayout)

        # Digit probabilities side
        digitGuesses = QWidget()
        digitGuesses.setStyleSheet("background-color: #a2cfdb; color: black; font-size: 18px;")
        digitsLayout = QVBoxLayout()

        self.digitsProbability = {}
        for i in range(10):
            self.digitsProbability[i] = QLabel(f"{i}:")
            digitsLayout.addWidget(self.digitsProbability[i])
        digitGuesses.setLayout(digitsLayout)

        self.mainLayout.addWidget(paintWidget, 2)
        self.mainLayout.addWidget(digitGuesses, 1)
        self.setCentralWidget(mainWidget)

    def getDigitTensor(self):
        imgTensor = self.canvas.convertToTensor()
        imgTensor = 255.0 - imgTensor
        _, rows, cols = imgTensor.nonzero(as_tuple=True)
        # Boundaries of the figure in the drawing
        minRow, maxRow = rows.min(), rows.max()
        minCol, maxCol = cols.min(), cols.max()
        croppedTensor = imgTensor[:, minRow: maxRow + 1, minCol: maxCol + 1]

        # Convert to MNIST format
        transform = tfs.Resize((28, 28))
        padding = torch.nn.ConstantPad2d(padding=16, value=0)
        resultTensor = padding(croppedTensor)
        resultTensor = transform(resultTensor).to(dtype=torch.uint8)
        torchvision.io.write_png(resultTensor, "../../../converted.png", 0)
        print("Saved")

    def save_canvas(self) -> None:
        """Save canvas to png with file dialog.

        Returns:
            None.
        """
        filePath, _ = QFileDialog.getSaveFileName(self, caption="caption", dir="../../..")
        filename = filePath.split("/")[-1]
        self.canvas.save(filePath)
        self.statusBar().showMessage(f"Canvas was saved to {filename}.png", timeout=1500)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
