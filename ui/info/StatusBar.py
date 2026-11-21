from PyQt6.QtWidgets import QStatusBar, QLabel, QSizePolicy

class StatusBar(QStatusBar):
    """Custom status bar for displaying messages"""
    def __init__(self, parent=None, name:str=""):
        super().__init__(parent)
        self.name_label = QLabel(name)
        self.addPermanentWidget(self.name_label)
        self.status_label = QLabel("")
        # Cache/analysis notices can be much wider than the window. Let the
        # available space constrain them instead of imposing a window minimum.
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.addWidget(self.status_label, 1)
    
    def update_name(self, name: str):
        self.name_label.setText(name)
    
    def update_status(self, status: str):
        self.status_label.setText(status)
        self.status_label.setToolTip(status)
