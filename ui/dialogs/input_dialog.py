import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFormLayout, QProgressBar

from ui.custom_widgets.styled_widgets import StyledDoubleSpinBox, StyledEdit, StyledComboBox
from utils.settings_handler import AppSettings


class CustomInputDialog(QWidget):
    def __init__(self, parent, title_name, question_name, placeholder="", min_width=300):
        super().__init__(parent)
        self.setWindowTitle(f"{title_name}")
        self.setWindowFlag(Qt.Tool)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.label = QLabel(f"{question_name}")
        self.edit = StyledEdit(theme=self.settings.read_theme())
        self.edit.setPlaceholderText(placeholder)

        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Ввести' if self.lang == 'RU' else "OK", self)

        btnLayout.addWidget(self.okBtn)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.edit)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)
        self.setMinimumWidth(min_width)

    def getText(self):
        return self.edit.text()


class CustomComboDialog(QWidget):
    def __init__(self, parent=None, theme='dark_blue.xml', dark_color=(255, 255, 255), light_color=(0, 0, 0),
                 title_name="ComboBoxTitle", question_name="Question:", variants=None, editable=False,
                 pre_label=None, post_label=None, width_percent=0.2, height_percent=0.1):
        super().__init__(parent)
        self.setWindowTitle(f"{title_name}")
        self.setWindowFlag(Qt.Tool)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        layout = QFormLayout()

        self.label = QLabel(f"{question_name}")
        self.combo = StyledComboBox(self, theme=theme, dark_color=dark_color, light_color=light_color)

        if variants:
            variants = np.array(variants)
        else:
            variants = np.array([""])

        self.combo.addItems(variants)

        self.combo.setEditable(editable)

        layout.addRow(self.label, self.combo)

        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Ввести' if self.lang == 'RU' else "OK", self)
        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else "Cancel", self)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        self.mainLayout = QVBoxLayout()

        if pre_label:
            self.mainLayout.addWidget(QLabel(pre_label))

        self.mainLayout.addLayout(layout)

        if post_label:
            self.mainLayout.addWidget(QLabel(post_label))

        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

        size, pos = self.settings.read_size_pos_settings()
        self.setMinimumWidth(int(size.width() * width_percent))
        self.setMinimumHeight(int(size.height() * height_percent))


    def getText(self):
        return self.combo.currentText()

    def hasText(self, text):
        for i in range(self.combo.count()):
            if self.combo.itemText(i) == text:
                return True

        return False

    def getPos(self, text):
        for i in range(self.combo.count()):
            if self.combo.itemText(i) == text:
                return i

        return None


class PromptInputDialog(QWidget):
    def __init__(self, parent, class_names=None, on_ok_clicked=None, prompts_variants=None, theme='dark_blue.xml',
                 dark_color=(255, 255, 255), light_color=(0, 0, 0), box_threshold=0.4, text_threshold=0.55):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.setWindowTitle("Выделение объектов по ключевым словам" if
                            self.lang == 'RU' else "Select objects by text prompt")
        self.setWindowFlag(Qt.Tool)

        prompt_layout = QFormLayout()

        self.prompt_label = QLabel("Что будем искать:" if self.lang == 'RU' else "Prompt:")
        self.prompt_combo = StyledComboBox(self, theme=theme, dark_color=dark_color, light_color=light_color)

        if prompts_variants:
            prompts_variants = np.array(prompts_variants)
        else:
            prompts_variants = np.array([""])

        self.prompt_combo.addItems(prompts_variants)
        self.prompt_combo.setEditable(True)

        prompt_layout.addRow(self.prompt_label, self.prompt_combo)

        class_layout = QFormLayout()

        self.class_label = QLabel("Каким классом разметить" if self.lang == 'RU' else "Select label name:")
        self.cls_combo = StyledComboBox(self, theme=theme, dark_color=dark_color, light_color=light_color)
        if not class_names:
            class_names = np.array(['no name'])
        self.cls_combo.addItems(np.array(class_names))
        self.cls_combo.setMinimumWidth(150)

        class_layout.addRow(self.class_label, self.cls_combo)

        classifier_layout = QFormLayout()

        self.box_threshold_spinbox = StyledDoubleSpinBox(self, theme=theme, dark_color=dark_color,
                                                         light_color=light_color)
        self.box_threshold_spinbox.setDecimals(3)
        self.box_threshold_spinbox.setValue(float(box_threshold))

        self.box_threshold_spinbox.setMinimum(0.01)
        self.box_threshold_spinbox.setMaximum(1.00)
        self.box_threshold_spinbox.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("IoU порог:" if self.lang == 'RU' else "IoU threshold"),
                                 self.box_threshold_spinbox)

        self.text_threshold_spinbox = StyledDoubleSpinBox(self, theme=theme, dark_color=dark_color,
                                                          light_color=light_color)
        self.text_threshold_spinbox.setDecimals(3)
        self.text_threshold_spinbox.setValue(float(text_threshold))

        self.text_threshold_spinbox.setMinimum(0.01)
        self.text_threshold_spinbox.setMaximum(1.00)
        self.text_threshold_spinbox.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("Текстовый порог:" if self.lang == 'RU' else "Text threshold"),
                                 self.text_threshold_spinbox)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Начать поиск' if self.lang == 'RU' else "Run", self)

        btnLayout.addWidget(self.okBtn)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(prompt_layout)
        self.mainLayout.addLayout(class_layout)
        self.mainLayout.addLayout(classifier_layout)
        self.mainLayout.addLayout(btnLayout)
        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

    def on_ok(self):
        self.prompt_combo.setVisible(False)
        self.prompt_label.setVisible(False)
        self.cls_combo.setVisible(False)
        self.class_label.setVisible(False)
        self.okBtn.setVisible(False)
        self.okBtn.setEnabled(False)

        self.on_ok_clicked()

    def getPrompt(self):
        return self.prompt_combo.currentText()

    def get_text_threshold(self):
        return self.text_threshold_spinbox.value()

    def get_box_threshold(self):
        return self.box_threshold_spinbox.value()

    def getClsName(self):
        return self.cls_combo.currentText()

    def getClsNumber(self):
        return self.cls_combo.currentIndex()

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)
