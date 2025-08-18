import torch
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QImage, QMouseEvent
import numpy as np

MIN_SIZE = QSize(300, 300)

class Canvas(QLabel):
    """Provides a widget for drawing figures on.

    Represents pixmap-pen pair
    """
    def __init__(self, parent=None):
        """Initialize canvas.

        Set parameters for Pixmap and Pen.

        Args:
            parent: The widget from which the canvas is inherited.
        """

        super().__init__(parent)
        self.setScaledContents(True)
        self.previous_point = None

        self.pixmap = QPixmap(MIN_SIZE)
        self.pixmap.fill(QColor("black"))
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())

        self.pen = QPen()
        self.pen.setColor(QColor("white"))
        self.pen.setWidth(12)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    def convertToTensor(self):
        pixmapImg = self.pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
        buf = pixmapImg.bits().tobytes()
        pixmapArray = (np.frombuffer(buf, dtype=np.uint8).reshape((1, self.pixmap.width(), self.pixmap.height())))
        return torch.tensor(pixmapArray)

    def save(self, filePath) -> None:
        """Save canvas to .png file.

        Args:
            filePath: Path to write a new file to.

        Returns:
            None.
        """
        self.pixmap.save(f"{filePath}.png", "PNG")

    def clear(self) -> None:
        """Clear canvas.

        Fill pixmap with white color.

        Returns:
            None.
        """
        self.pixmap.fill(QColor("black"))
        self.setPixmap(self.pixmap)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse left button click.

        Draw a point on a pixmap by clicking using QPainter.
        Every two consecutive points are connected by drawing segment in order to smooth painting.

        Args:
            event: Mouse click event properties.

        Returns:
            None
        """
        cursorPos = event.position().toPoint()
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self.pen)
        # Connect successive points with line
        if self.previous_point:
            painter.drawLine(self.previous_point, cursorPos)
        else:
            painter.drawPoint(cursorPos)
        painter.end()
        # Update pixmap
        self.setPixmap(self.pixmap)
        self.previous_point = cursorPos

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse left button release.

        Reset the position of the last drawn point to disable connection to the next one.

        Args:
            event: Mouse click event properties.

        Returns:
            None
        """
        self.previous_point = None
