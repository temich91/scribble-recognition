import sys
import torch
import torchvision.transforms as tfs
from PySide6.QtWidgets import (QMainWindow, QApplication, QWidget, QSizePolicy, QProgressBar,
                               QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog)
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import Qt
from canvas import Canvas
from title_bar import TitleBar

MINIMAL_SIZE = QSize(500, 380)

class MainWindow(QMainWindow):
    """Main window class.
    """
    def __init__(self):
        """ Initialize window properties and layout.

        """
        super().__init__()

        # self.model = ConvNet().load_state_dict(torch.load("../model/models/cnn.pt", weights_only=True))

        # Window setup
        self.setMinimumSize(MINIMAL_SIZE)
        self.setWindowTitle("ScribbleRecognizer")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.statusBar()

        mainWidget = QWidget()

        # Digit drawing side
        paintWidget = QWidget()

        self.canvas = Canvas(parent=paintWidget)
        self.canvas.setObjectName("canvas")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.canvas.changed.connect(self.predictProbas)

        # Buttons for paint actions/options
        self.saveBtn = QPushButton("&Save")
        self.saveBtn.clicked.connect(self.saveCanvas)
        # self.saveBtn.clicked.connect(self.getDigitTensor)

        self.clearBtn = QPushButton("Clear")
        # self.clearBtn.clicked.connect(self.canvas.clear)
        self.clearBtn.clicked.connect(self.setRandomProbs)

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
        digitsLayout = QVBoxLayout()

        self.digitsProbability = {}
        for i in range(10):
            self.digitsProbability[i] = QHBoxLayout()
            self.digitsProbability[i].addWidget(QLabel(f"{i}"))
            self.digitsProbability[i].addWidget(QProgressBar())
            digitsLayout.addLayout(self.digitsProbability[i])
        digitGuesses.setLayout(digitsLayout)

        self.contentLayout = QHBoxLayout()
        self.contentLayout.addWidget(paintWidget, 2)
        self.contentLayout.addWidget(digitGuesses, 1)

        self.mainLayout = QVBoxLayout(mainWidget)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.titleBar = TitleBar()
        self.titleBar.minimizeBtn.clicked.connect(self.minimize)
        self.titleBar.maximizeBtn.clicked.connect(self.maximize)
        self.titleBar.closeBtn.clicked.connect(self.close)

        self.mainLayout.addWidget(self.titleBar)
        self.mainLayout.addLayout(self.contentLayout)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(mainWidget)

    def minimize(self):
        self.titleBar.maximizeBtn.setText("🗖")
        self.showMinimized()

    def maximize(self):
        self.titleBar.maximizeBtn.setText("🗗")
        self.showMaximized()

    def setRandomProbs(self):
        import random
        for i in range(10):
            self.digitsProbability[i].itemAt(1).widget().setValue(random.random() * 100)

    def getDigitTensor(self):
        imgTensor = self.canvas.convertToTensor()
        _, rows, cols = imgTensor.nonzero(as_tuple=True)
        # Boundaries of the figure in the drawing
        minRow, maxRow = rows.min(), rows.max()
        minCol, maxCol = cols.min(), cols.max()
        croppedTensor = imgTensor[:, minRow: maxRow + 1, minCol: maxCol + 1]

        # Convert to MNIST format
        transform = tfs.Resize((28, 28))
        padding = torch.nn.ConstantPad2d(padding=16, value=0)
        resultTensor = padding(croppedTensor)
        return transform(resultTensor).to(dtype=torch.uint8)

    def predictProbas(self):
        data = self.getDigitTensor()
        pred = self.model(data)
        print(pred)

    def saveCanvas(self) -> None:
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
    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    window = MainWindow()
    window.show()
    app.exec()
