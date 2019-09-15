from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QSize

from .game_view import GameView


class Visualization(QMainWindow):
    def __init__(self, game_view: GameView, refresh_views: bool = True):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(640, 480))
        self.setWindowTitle("Bughouse visualization")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        lay = QVBoxLayout(central_widget)

        self._label = QLabel(self)

        lay.addWidget(self._label)

        self._image = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        self._game_view = game_view

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self._update_timer)
        timer.setSingleShot(False)
        timer.start(100)
        self.update_board()
        self._refresh_views = refresh_views

    def _update_timer(self):
        if self._refresh_views:
            while self._game_view.updates_available():
                self._game_view.receive_updates()
        self.update_board()

    def update_board(self) -> None:
        self._image.fill(0)
        painter = QPainter()
        painter.begin(self._image)
        renderer = QSvgRenderer()
        renderer.load(self._game_view._repr_svg_().encode("utf-8"))
        renderer.render(painter)
        painter.end()
        self._label.setPixmap(QPixmap.fromImage(self._image))
        self.setMinimumSize(QSize(self._image.width() + 10, self._image.height() + 10))
