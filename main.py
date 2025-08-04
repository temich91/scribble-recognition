import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QSizePolicy, QLabel, QFileDialog, QVBoxLayout
from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap, Qt, QPainter, QPen, QGradient, QColor

MINIMAL_SIZE = QSize(400, 400)

class Painter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(MINIMAL_SIZE)
        self.setWindowTitle("DigitPaint")

        self.main_label = QLabel()
        self.main_label.setScaledContents(True)
        self.canvas = QPixmap(MINIMAL_SIZE)
        self.canvas.fill(QColor("white"))

        self.pen = QPen()
        self.pen.setColor(QColor("black"))
        self.pen.setWidth(4)
        self.pen.setCapStyle(Qt.PenCapStyle.SquareCap)

        self.previousPoint = None

        self.main_label.setPixmap(self.canvas)
        self.main_label.resize(self.canvas.width(), self.canvas.height())
        self.setCentralWidget(self.main_label)

    def resizeEvent(self, event):
        self.canvas = self.canvas.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding)

    def mouseMoveEvent(self, event):
        cursor_pos = event.position()
        painter = QPainter(self.canvas)
        painter.setPen(self.pen)
        if self.previousPoint:
            painter.drawLine(self.previousPoint.x(), self.previousPoint.y(), cursor_pos.x(), cursor_pos.y())
        else:
            painter.drawPoint(int(cursor_pos.x()), int(cursor_pos.y()))
        painter.end()

        self.previousPoint = cursor_pos
        self.main_label.setPixmap(self.canvas)

    def mouseReleaseEvent(self, event):
        self.previousPoint = None

app = QApplication(sys.argv)
window = Painter()
window.show()
app.exec()
