import os
import math
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QFormLayout, QApplication

from ui.custom_widgets.card import Card
from ui.custom_widgets.styled_widgets import StyledSpinBox
from ui.dialogs.export_steps.preprocess_blocks.preprocess_block import PreprocessBlock
from utils.settings_handler import AppSettings

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class TileCard(Card):
    """
    Карточка этапа обработки "Tiles".
    """

    def __init__(self, test_image_path, min_width=300, min_height=150, max_width=100, on_ok=None, on_delete=None):

        self.icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_icons')

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        name = "Tile" if self.lang == 'RU' else 'Tile'

        if self.lang == "RU":
            sub_text = "Разрезать изображения"
        else:
            sub_text = "Tile images"

        self.on_ok = on_ok
        self.test_image_path = test_image_path

        super(TileCard, self).__init__(None, text=name,
                                       path_to_img=os.path.join(self.icons_path, 'tile.png'),
                                       min_width=min_width, min_height=min_height, max_width=max_width,
                                       on_edit_clicked=self.on_tile_edit, is_del_button=True, is_edit_button=True,
                                       on_del=on_delete, sub_text=sub_text)

        self.new_size = None

    def get_new_size(self):
        """
        Возвращает новый размер изображений
        """
        return self.new_size

    def on_tile_edit(self):
        """
        При нажатии на кнопку Edit карточки "Resize"
        Открытие окна редактирования размера изображений
        """

        self.resize_widget = TileStep(self.test_image_path, on_ok=self.on_tile_ok)
        self.resize_widget.show()

    def on_tile_ok(self):

        self.new_size = self.resize_widget.get_options()

        if self.lang == 'RU':
            sub_text = f"Размер изображений: {self.new_size}"
        else:
            sub_text = f"Images size: {self.new_size}"

        self.set_sub_text(sub_text)

        self.resize_widget.hide()

        if self.on_ok:
            # Если в конструкторе передан хук
            self.on_ok()


class TileStep(PreprocessBlock):
    """
    Окно редактирования размера изображений.
    """

    def __init__(self, test_image_path, min_width=1280, minimum_expand_height=760, on_ok=None,
                 on_cancel=None, thumbnail_size=640, tile_width=800, tile_height=600, overlap_percent=30):
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.test_img_path = test_image_path
        name = "tile"
        if self.lang == "RU":
            title = "Разрезать изображения"
        else:
            title = "Tile images"

        super().__init__(name, options=None, parent=None, title=title, min_width=min_width,
                         minimum_expand_height=minimum_expand_height, icon_name="tile.png")

        self.img = Image.open(test_image_path)
        self.img_width, self.img_height = self.img.size

        print(f'Img width {self.img_width} height {self.img_height}')
        self.img = QImage(test_image_path)

        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap_percent = overlap_percent

        self.scale = 1.0 * self.img_width / thumbnail_size
        self.thumbnail_size = thumbnail_size

        self.thumbnail_width = thumbnail_size
        self.thumbnail_height = int(self.img_height / self.scale)

        self.create_inputs_layout()
        self.create_images_layout(self.tile_width, self.tile_height)

        self.create_buttons(on_ok, on_cancel)
        self.setMinimumWidth(min_width)

    def draw_grid(self, width, height):
        # Create a painter
        painter = QPainter(self.pix_resized)

        # Set pen and brush
        red_pen = QPen(QColor(255, 0, 0), 2)  # Red color with width 2
        red_pen.setStyle(Qt.DashLine)
        green_pen = QPen(QColor(0, 255, 0, 50), 1)  # Red color with width 2
        painter.setPen(red_pen)

        brush = QBrush()
        brush.setColor(QColor(0, 255, 0, 50))
        brush.setStyle(Qt.Dense1Pattern)
        painter.setBrush(brush)

        # Draw grid lines
        # horizontal
        num_of_hor_lines = math.ceil(self.img_height / height)
        for i in range(1, num_of_hor_lines):  # Adjust the number of rows and columns as needed
            if i * int(height / self.scale) == self.pix_resized.height():
                continue

            painter.setPen(red_pen)
            painter.drawLine(0, i * int(height / self.scale), self.pix_resized.width(),
                             i * int(height / self.scale))

            painter.setPen(green_pen)
            delta = self.overlap_spin.value() * height / (self.scale * 100.0)
            y1 = i * (int(height / self.scale)) - int(delta)
            painter.drawRect(0, y1, self.pix_resized.width(),
                             int(delta) * 2)

        # vertical
        num_of_ver_lines = math.ceil(self.img_width / width)
        for i in range(1, num_of_ver_lines):  # Adjust the number of rows and columns as needed
            if i * int(width / self.scale) == self.pix_resized.width():
                continue
            painter.setPen(red_pen)
            painter.drawLine(i * int(width / self.scale), 0,
                             i * int(width / self.scale), self.pix_resized.height())

            painter.setPen(green_pen)
            delta = self.overlap_spin.value() * width / (self.scale * 100.0)
            x1 = i * (int(width / self.scale)) - int(delta)
            painter.drawRect(x1, 0,
                             2 * int(delta), self.pix_resized.height())

    def create_images_layout(self, tile_width, tile_height):
        self.images_layout = QHBoxLayout()
        self.pix_original = QPixmap(self.test_img_path)
        self.pix_original = self.pix_original.scaled(self.thumbnail_width, self.thumbnail_height)

        self.label_img_original = QLabel()
        self.label_img_original.setPixmap(self.pix_original)
        self.pix_resized = QPixmap.fromImage(self.img)
        self.pix_resized = self.pix_resized.scaled(self.thumbnail_width, self.thumbnail_height)

        self.draw_grid(tile_width, tile_height)

        self.label_img_resized = QLabel()
        self.label_img_resized.setPixmap(self.pix_resized)
        self.images_layout.addWidget(self.label_img_original)
        self.images_layout.addWidget(self.label_img_resized)
        self.layout.addLayout(self.images_layout)

    def create_inputs_layout(self):
        self.inputs_layout = QVBoxLayout()
        header_text = "Введите размер плитки" if self.lang == 'RU' else "Input tile size"
        self.header = QLabel(header_text)
        self.header.setAlignment(Qt.AlignVCenter)

        form_lay = QFormLayout()
        width_text = "Ширина, px" if self.lang == 'RU' else "Width, px"
        height_text = "Высота, px" if self.lang == 'RU' else "Height, px"
        overlap_text = "Процент перекрытия" if self.lang == 'RU' else "Overlap percent"
        self.width_spin = StyledSpinBox(theme=self.settings.read_theme())

        self.width_spin.setMinimum(16)
        self.width_spin.setMaximum(10000)

        self.width_spin.setValue(self.tile_width)
        self.width_spin.valueChanged.connect(self.on_width_change)

        self.height_spin = StyledSpinBox(theme=self.settings.read_theme())

        self.height_spin.setMinimum(16)
        self.height_spin.setMaximum(10000)

        self.height_spin.setValue(self.tile_height)
        self.height_spin.valueChanged.connect(self.on_height_change)

        self.overlap_spin = StyledSpinBox(theme=self.settings.read_theme())

        self.overlap_spin.setMinimum(0)
        self.overlap_spin.setMaximum(100)

        self.overlap_spin.setValue(self.overlap_percent)
        self.overlap_spin.valueChanged.connect(self.on_overlap_spin_change)

        form_lay.addRow(QLabel(width_text), self.width_spin)
        form_lay.addRow(QLabel(height_text), self.height_spin)
        form_lay.addRow(QLabel(overlap_text), self.overlap_spin)

        self.inputs_layout.addWidget(self.header)
        self.inputs_layout.addLayout(form_lay)

        self.layout.addLayout(self.inputs_layout)

    def on_overlap_spin_change(self, value):
        self.fit_thumbnail(self.width_spin.value(), self.height_spin.value())

    def on_width_change(self, new_width):
        self.fit_thumbnail(new_width, self.tile_height)

    def fit_thumbnail(self, new_width, new_height):

        self.width_spin.setValue(new_width)
        self.height_spin.setValue(new_height)

        self.images_layout.removeWidget(self.label_img_resized)
        self.pix_resized = QPixmap.fromImage(self.img)
        self.pix_resized = self.pix_resized.scaled(self.thumbnail_width, self.thumbnail_height)

        self.draw_grid(new_width, new_height)

        self.label_img_resized = QLabel()
        self.label_img_resized.setPixmap(self.pix_resized)
        self.images_layout.addWidget(self.label_img_resized)

    def on_height_change(self, new_height):

        self.fit_thumbnail(self.tile_width, new_height)

    def get_options(self):
        return self.width_spin.value(), self.height_spin.value()


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    from PyQt5.QtWidgets import QApplication


    def on_ok_test():
        print(w.get_new_size())


    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    image_path = "cat.jpg"

    w = TileCard(image_path, on_ok=on_ok_test)

    w.show()

    app.exec_()
