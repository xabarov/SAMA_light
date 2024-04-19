from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon

from utils.settings_handler import AppSettings


class ImagesPanel(QWidget):
    def __init__(self, parent, add_image_to_projectAct, del_image_from_projectAct, change_image_status_act,
                 icon_folder, button_size=30,
                 on_color_change_signal=None, on_images_list_change=None):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # Панель изображений - заголовок
        images_header = QHBoxLayout()
        images_header.addWidget(QLabel("Список изображений" if self.lang == 'RU' else "Images list"))

        self.add_im_button = QPushButton()
        self.add_im_button.clicked.connect(add_image_to_projectAct)
        images_header.addWidget(self.add_im_button)

        self.change_im_status_button = QPushButton()
        self.change_im_status_button.clicked.connect(change_image_status_act)
        images_header.addWidget(self.change_im_status_button)

        self.del_im_button = QPushButton()
        self.del_im_button.clicked.connect(del_image_from_projectAct)
        images_header.addWidget(self.del_im_button)

        self.add_im_button.setIcon((QIcon(icon_folder + "/add.png")))
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        self.change_im_status_button.setIcon((QIcon(icon_folder + "/clipboard.png")))

        self.add_im_button.setFixedHeight(button_size)
        self.del_im_button.setFixedHeight(button_size)
        self.change_im_status_button.setFixedHeight(button_size)

        self.add_im_button.setFixedWidth(button_size)
        self.del_im_button.setFixedWidth(button_size)
        self.change_im_status_button.setFixedWidth(button_size)

        self.add_im_button.setStyleSheet("border: none;")
        self.del_im_button.setStyleSheet("border: none;")
        self.change_im_status_button.setStyleSheet("border: none;")

        if on_color_change_signal:
            on_color_change_signal.connect(self.on_color_change)

        if on_images_list_change:
            on_images_list_change.connect(self.on_im_list_change)

        self.setLayout(images_header)

        self.toggle_buttons(False)

    def on_color_change(self, icon_folder):
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        self.add_im_button.setIcon((QIcon(icon_folder + "/add.png")))
        self.change_im_status_button.setIcon((QIcon(icon_folder + "/clipboard.png")))

    def on_im_list_change(self, images_size):
        if images_size > 0:
            self.toggle_buttons(True)
            return
        self.toggle_buttons(False)

    def toggle_buttons(self, enable):
        self.del_im_button.setEnabled(enable)
        self.add_im_button.setEnabled(enable)
        self.change_im_status_button.setEnabled(enable)


class LabelsPanel(QWidget):
    def __init__(self, parent, del_label_from_image_act, clear_all_labels, icon_folder, button_size=30,
                 on_color_change_signal=None,
                 on_labels_count_change=None, on_selected_num_changed=None):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # Панель изображений - заголовок
        self.header = QHBoxLayout()
        self.header.addWidget(QLabel("Список меток" if self.lang == 'RU' else "Labels list"))
        self.del_label_from_image_act = del_label_from_image_act
        self.on_color_change_signal = on_color_change_signal
        self.on_labels_count_change = on_labels_count_change
        self.on_selected_num_changed = on_selected_num_changed
        self.button_size = button_size
        self.icon_folder = icon_folder
        self.clear_all_labels = clear_all_labels

        # Del label
        self.create_del_label()

        # Clean labels
        self.create_clean_button()

        if self.on_color_change_signal:
            self.on_color_change_signal.connect(self.on_color_change)

        if self.on_labels_count_change:
            self.on_labels_count_change.connect(self.on_lbs_count_change)

        if self.on_selected_num_changed:
            self.on_selected_num_changed.connect(self.on_sel_num_changed)

        self.setLayout(self.header)

    def on_sel_num_changed(self, count):
        if count > 0:
            self.del_im_button.setEnabled(True)
        else:
            self.del_im_button.setEnabled(False)

    def create_del_label(self):
        self.del_im_button = QPushButton()
        self.del_im_button.setEnabled(False)
        self.del_im_button.clicked.connect(self.del_label_from_image_act)
        self.header.addWidget(self.del_im_button)

        self.del_im_button.setIcon((QIcon(self.icon_folder + "/del.png")))

        self.del_im_button.setFixedHeight(self.button_size)
        self.del_im_button.setFixedWidth(self.button_size)

        self.del_im_button.setStyleSheet("border: none;")

    def create_clean_button(self):
        self.clean_button = QPushButton()
        self.clean_button.setEnabled(False)
        self.clean_button.clicked.connect(self.clear_all_labels)
        self.header.addWidget(self.clean_button)

        self.clean_button.setIcon((QIcon(self.icon_folder + "/clean.png")))

        self.clean_button.setFixedHeight(self.button_size)

        self.clean_button.setFixedWidth(self.button_size)

        self.clean_button.setStyleSheet("border: none;")

    def on_color_change(self, icon_folder):
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        self.clean_button.setIcon((QIcon(icon_folder + "/clean.png")))

    def on_lbs_count_change(self, count):
        if count > 0:
            self.clean_button.setEnabled(True)
        else:
            self.clean_button.setEnabled(False)
