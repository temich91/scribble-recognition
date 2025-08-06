from fileinput import filename

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QSize, Qt, QIODevice, QBuffer
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen

MIN_SIZE = QSize(300, 300)

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)
        self.previous_point = None

        self.pixmap = QPixmap(MIN_SIZE)
        self.pixmap.fill(QColor("white"))
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())

        self.pen = QPen()
        self.pen.setColor(QColor("black"))
        self.pen.setWidth(12)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    def save(self, filePath):
        self.pixmap.save(f"{filePath}.png", "PNG")

    def clear(self):
        self.pixmap.fill(QColor("white"))
        self.setPixmap(self.pixmap)

    def mouseMoveEvent(self, event):
        cursor_pos = event.position().toPoint()
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self.pen)
        if self.previous_point:
            painter.drawLine(self.previous_point, cursor_pos)
        else:
            painter.drawPoint(cursor_pos)
        painter.end()

        self.setPixmap(self.pixmap)
        self.previous_point = cursor_pos

    def mouseReleaseEvent(self, event):
        self.previous_point = None
