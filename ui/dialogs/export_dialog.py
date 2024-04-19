import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QApplication, QMessageBox

from ui.custom_widgets.accordion import Accordion
from ui.custom_widgets.enumerate_card import EnumerateCard
from ui.dialogs.export_steps.preprocess_step import PreprocessStep
from ui.dialogs.export_steps.set_export_path_widget import SetPathWidget
from ui.dialogs.export_steps.train_test_splitter import TrainTestSplitter
from utils.settings_handler import AppSettings


class ExportDialog(QWidget):
    def __init__(self, test_image_path, width=800, height=200, on_ok_clicked=None, label_names=None):
        """
        Экспорт датасета
        """
        super().__init__(None)
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.setWindowTitle("Экспорт датасета" if self.lang == 'RU' else "Export dataset")
        self.setWindowFlag(Qt.Tool)
        self.set_width = width
        self.set_height = height

        self.cards = []

        # STEP 1. Path

        self.export_path_step = SetPathWidget(None, self.settings.read_theme())

        export_path_title = "Формат/Путь" if self.lang == 'RU' else "Format/Path"
        export_card = EnumerateCard(body=self.export_path_step, num=1, text=export_path_title, is_number_flat=True)
        self.cards.append(export_card)
        # STEP 2. Splitter

        self.labels_splitter = TrainTestSplitter(None)
        splitter_title = "Train/Val/Test"
        splitter_card = EnumerateCard(body=self.labels_splitter, num=2, text=splitter_title, is_number_flat=True)
        self.cards.append(splitter_card)

        # STEP 3. Preprocess
        self.preprocess_step = PreprocessStep(labels=label_names, test_image_path=test_image_path)
        preprocess_title = "Предобработка" if self.lang == 'RU' else "Preprocess"
        preprocess_card = EnumerateCard(body=self.preprocess_step, num=3, text=preprocess_title, is_number_flat=True)
        self.cards.append(preprocess_card)
        # Разбиение на train, val

        self.accordion = Accordion(self.cards)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        # Buttons layout:
        self.okBtn = QPushButton('Экспортировать' if self.lang == 'RU' else "Export", self)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)
        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.okBtn)
        button_layout.addWidget(self.cancelBtn)

        # Stack layers

        self.layout = QVBoxLayout()

        self.layout.addWidget(self.accordion)
        self.layout.addWidget(self.progress_bar)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        self.data = {}

        self.resize(int(width), int(height))

    def get_splits(self):
        return self.labels_splitter.get_splits()

    def get_idx_text_variant(self):
        return self.labels_splitter.get_idx_text_variant()

    def get_idx_text_sim(self):
        return self.labels_splitter.get_idx_text_sim()

    def get_export_format(self):
        return self.export_path_step.get_export_format()

    def get_new_image_size(self):
        params = self.preprocess_step.get_params()
        if "resize" in params:
            return params["resize"]

    def on_choose_labels_checkbox_clicked(self):
        is_checked = self.choose_labels_checkbox.isChecked()
        self.export_labels_list.setVisible(is_checked)

        self.adjustSize()
        self.setMinimumWidth(self.set_width)

    def get_export_path(self):
        return self.export_path_step.get_export_path()

    def get_labels_map(self):
        params = self.preprocess_step.get_params()
        if "modify_classes" in params:
            return params["modify_classes"]

    def is_filter_null(self):
        params = self.preprocess_step.get_params()
        if "filter_null" in params:
            return True

    def show_empty_path_message(self):
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Information)

        if self.lang == 'RU':
            text = "Укажите имя директории для экспорта"
        else:
            text = "Please set export folder name"

        msgbox.setText(text)
        msgbox.setWindowTitle("Внимание!" if self.lang == 'RU' else "Attention!")
        msgbox.exec()

    def on_ok(self):
        if self.get_export_path() != "":
            for card in self.cards:
                card.setEnabled(False)
                card.setVisible(False)

            self.okBtn.setVisible(False)
            self.cancelBtn.setVisible(False)

            self.adjustSize()
            self.setMinimumWidth(self.set_width)

            self.on_ok_clicked()
        else:
            self.show_empty_path_message()

    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)


if __name__ == '__main__':
    def on_ok():
        print("Ok!")


    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    labels = ['F-16', 'F-35', 'C-130', 'C-17']
    dialog = ExportDialog(label_names=labels, on_ok_clicked=on_ok, theme='dark_blue.xml')
    dialog.show()
    sys.exit(app.exec_())
