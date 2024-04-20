import os
import sys

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon, QCursor, QKeySequence
from PyQt5.QtWidgets import QAction, QMenu
from qt_material import apply_stylesheet

import utils.help_functions as hf
from ui.base_window import MainWindow
from utils import config
from utils import ml_config
from utils.sam.fast_sam_worker import FastSAMWorker
from utils.states import DrawState


class AnnotatorSAM(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Annotator SAM")

        # SAM
        self.image_set = False
        self.fast_sam_worker = None
        self.queue_to_fast_sam_worker = []

        self.view.mask_end_drawing.on_mask_end_drawing.connect(self.ai_mask_end_drawing)

        self.handle_sam_model()

    def handle_sam_model(self):
        """
        Загрузка модели SAM
        """
        self.fast_sam_worker = FastSAMWorker()
        self.fast_sam_worker.finished.connect(self.on_image_set)
        if self.tek_image_path:
            self.queue_image_to_sam(self.tek_image_path)

    def reset_shortcuts(self):

        super(AnnotatorSAM, self).reset_shortcuts()

        shortcuts = self.settings.read_shortcuts()

        for sc, act in zip(
                ['sam_box', 'sam_points'],
                [self.aiAnnotatorMaskAct, self.aiAnnotatorPointsAct]):
            shortcut = shortcuts[sc]
            appearance = shortcut['appearance']
            act.setShortcut(QKeySequence(appearance))

    def ai_mask_end_drawing(self):
        """
        Завершение рисования прямоугольной области SAM
        """
        self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))
        input_box = self.view.get_sam_mask_input()

        self.view.remove_items_from_active_group()

        if len(input_box):
            if self.image_set and not self.fast_sam_worker.isRunning():
                shapes = self.fast_sam_worker.box(input_box)
                self.add_sam_shapes_to_view(shapes)

        self.view.end_drawing()
        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

        self.save_view_to_project()

    def createActions(self):
        """
        Добавляем новые действия к базовой модели
        """

        # AI Annotators
        self.aiAnnotatorPointsAct = QAction(
            "Сегментация по точкам" if self.lang == 'RU' else "SAM by points",
            self, enabled=False,
            triggered=self.ai_points_pressed,
            checkable=True)
        self.aiAnnotatorMaskAct = QAction(
            "Сегментация внутри бокса" if self.lang == 'RU' else "SAM by box", self,
            enabled=False,
            triggered=self.ai_mask_pressed,
            checkable=True)

        super(AnnotatorSAM, self).createActions()

    def toggle_act(self, is_active):
        """
        Переключение действий, зависящих от состояния is_active
        """
        super(AnnotatorSAM, self).toggle_act(is_active)
        self.aiAnnotatorMethodMenu.setEnabled(is_active)
        self.aiAnnotatorPointsAct.setEnabled(is_active)
        self.aiAnnotatorMaskAct.setEnabled(is_active)

    def createMenus(self):
        super(AnnotatorSAM, self).createMenus()

        self.aiAnnotatorMethodMenu = QMenu("С помощью ИИ" if self.lang == 'RU' else "AI", self)

        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorPointsAct)
        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorMaskAct)

        self.AnnotatorMethodMenu.addMenu(self.aiAnnotatorMethodMenu)

        self.menuBar().clear()
        for menu in [self.fileMenu, self.viewMenu, self.annotatorMenu, self.datasetMenu,
                     self.settingsMenu,
                     self.helpMenu]:
            self.menuBar().addMenu(menu)

    def set_icons(self):
        super(AnnotatorSAM, self).set_icons()
        # AI
        self.aiAnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/ai.png"))
        self.aiAnnotatorPointsAct.setIcon(QIcon(self.icon_folder + "/mouse.png"))
        self.aiAnnotatorMaskAct.setIcon(QIcon(self.icon_folder + "/ai_select.png"))

    def open_image(self, image_name):
        """
        К базовой модели добавляется SAM. Поскольку ему нужен прогрев, создаем очередь загружаемых изображений
        """
        super(AnnotatorSAM, self).open_image(image_name)

        self.image_set = False
        self.queue_image_to_sam(image_name)  # создаем очередь загружаемых изображений

    def reload_image(self, is_tek_image_changed=False):
        """
        Заново загружает текущее изображение с разметкой
        """
        super(AnnotatorSAM, self).reload_image(is_tek_image_changed=is_tek_image_changed)
        self.view.clear_ai_points()  # очищаем точки-prompts SAM

    def set_image(self, image_name):
        """
        Старт загрузки изображения в модель SAM
        """
        self.cv2_image = cv2.imread(image_name)
        self.image_set = False
        self.fast_sam_worker.set_image(self.cv2_image)
        self.queue_to_fast_sam_worker = []
        self.info_message(
            "Нейросеть SAM еще не готова. Подождите секунду..." if self.lang == 'RU' else "SAM is loading. Please "
                                                                                          "wait...")
        self.fast_sam_worker.start()

    def get_jpg_path(self, image_name):
        """
        Для поддержки детектора. У него данный метод переписан и поддерживает конвертацию в tif
        """
        return image_name

    def on_image_set(self):
        """
        Завершение прогрева модели SAM. Если остались изображения в очереди - берем последнее, а очередь очищаем
        """

        if len(self.queue_to_fast_sam_worker) != 0:
            image_name = self.queue_to_fast_sam_worker[-1]  # geo_tif_names
            jpg_path = self.get_jpg_path(image_name)
            self.set_image(jpg_path)

        else:
            self.info_message(
                "Нейросеть SAM готова к сегментации" if self.lang == 'RU' else "SAM ready to work")
            self.image_set = True

    def ai_points_pressed(self):
        """
        Нажатие левой или правой кнопки мыши в режиме точек-prompts SAM
        """
        self.draw_state = DrawState.ai_points
        self.start_view_drawing(self.draw_state)

    def ai_mask_pressed(self):
        """
        Старт рисования прямоугольной области в режиме SAM
        """
        self.draw_state = DrawState.ai_mask
        self.start_view_drawing(self.draw_state)

    def add_sam_shapes_to_view(self, shapes):
        cls_num = self.cls_combo.currentIndex()
        alpha_tek = self.settings.read_alpha()
        alpha_edge = self.settings.read_edges_alpha()

        points_mass = []

        for shape in shapes:
            points = shape['points']
            points_mass.append(points)

        color = None
        label = self.project_data.get_label_name(cls_num)
        if label:
            color = self.project_data.get_label_color(label)
        if not color:
            color = ml_config.PALETTE[cls_num]

        cls_name = self.cls_combo.itemText(cls_num)

        label_text_params = self.settings.read_label_text_params()
        if label_text_params['hide']:
            text = None
        else:
            text = f"{cls_name}"

        self.view.add_polygons_group_to_scene(cls_num, points_mass, color=color, text=text,
                                              alpha=alpha_tek,
                                              alpha_edge=alpha_edge)

    def start_drawing(self):
        """
        Старт рисования метки
        """
        super(AnnotatorSAM, self).start_drawing()
        self.view.clear_ai_points()  # очищение точек SAM

    def break_drawing(self):
        """
        Прерывание рисования метки
        """
        super(AnnotatorSAM, self).break_drawing()
        if self.draw_state == DrawState.ai_points:
            self.view.clear_ai_points()
            self.view.remove_items_from_active_group()

    def end_drawing(self):
        """
        Завершение рисования метки
        """
        super(AnnotatorSAM, self).end_drawing()

        if self.draw_state == DrawState.ai_points:

            self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

            input_point, input_label = self.view.get_sam_input_points_and_labels()

            if len(input_label):
                if self.image_set and not self.fast_sam_worker.isRunning():
                    shapes = self.fast_sam_worker.point(input_point, input_label)
                    self.add_sam_shapes_to_view(shapes)

            else:
                self.view.remove_items_from_active_group()

            self.view.end_drawing()  # clear points inside view

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

            self.save_view_to_project()

    def on_quit(self):
        """
        Выход из приложения
        """
        self.exit_box.hide()

        self.write_size_pos()

        self.hide()  # Скрываем окно

        if self.fast_sam_worker:
            self.fast_sam_worker.running = False  # Изменяем флаг выполнения
            self.fast_sam_worker.wait(5000)  # Даем время, чтобы закончить

        self.is_asked_before_close = True
        self.close()

    def queue_image_to_sam(self, image_name):
        """
        Постановка в очередь изображения для загрузки в модель SAM
        """
        if not self.fast_sam_worker.isRunning():
            self.fast_sam_worker.set_image(self.cv2_image)
            self.info_message(
                "Начинаю загружать изображение в нейросеть SAM..." if self.lang == 'RU' else "Start loading image to "
                                                                                             "SAM...")
            self.fast_sam_worker.start()
        else:
            self.queue_to_fast_sam_worker.append(image_name)
            if self.lang == 'RU':
                message = f"Изображение {os.path.split(image_name)[-1]} добавлено в очередь на обработку."
            else:
                message = f"Image {os.path.split(image_name)[-1]} is added to queue..."
            self.info_message(message)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = AnnotatorSAM()
    w.show()
    sys.exit(app.exec_())
