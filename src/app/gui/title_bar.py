from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QPoint

class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dragging = False
        self.drag_position = QPoint()

        self.layout = QHBoxLayout(self)
        self.setFixedHeight(40)
        self.setObjectName("title_bar")

        self.closeBtn = QPushButton("⨉")
        self.maximizeBtn = QPushButton("🗖")
        self.minimizeBtn = QPushButton("🗕")
        self.layout.addStretch()
        self.layout.addWidget(self.minimizeBtn)
        self.layout.addWidget(self.maximizeBtn)
        self.layout.addWidget(self.closeBtn)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.dragging:
            self.window().move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
