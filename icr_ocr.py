from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from pyqtgraph import TextItem
import time
import sys
import random

class LivePlot(QWidget):
    def __init__(self, max_points=200, update_interval_ms=250, parent=None):
        super().__init__(parent)

        self.max_points = max_points
        self.x_data = []
        self.y1_data = []
        self.y2_data = []
        self.start_time = None

        self.setMaximumHeight(256)
        self.layout = QVBoxLayout(self)        
        self.plot_widget = pg.PlotWidget(title="Real-Time ICR/OCR")
        self.layout.addWidget(self.plot_widget)

        self.plot_widget.addLegend(offset=(10, 10))
        self.curve1 = self.plot_widget.plot(pen='g', name="OCR")
        self.curve2 = self.plot_widget.plot(pen='m', name="ICR")
        self.label1 = None
        self.label2 = None
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setMouseEnabled(x=False, y=False)

        self.new_y1 = None
        self.new_y2 = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(update_interval_ms)

    def add_data_point(self, y1, y2):
        self.new_y1 = y1
        self.new_y2 = y2

    def reset_plot(self):
        self.start_time = None
        self.x_data = []
        self.y1_data = []
        self.y2_data = []
        if self.label1:
            self.plot_widget.removeItem(self.label1)
            self.label1 = None
        if self.label2:
            self.plot_widget.removeItem(self.label2)
            self.label2 = None
        self.curve1.clear()
        self.curve2.clear()
        self.new_y1 = None
        self.new_y2 = None

    def refresh_plot(self):
        if self.new_y1 is None or self.new_y2 is None:
            return

        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        self.x_data.append(elapsed_time)
        self.y1_data.append(self.new_y1)
        self.y2_data.append(self.new_y2)

        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y1_data = self.y1_data[-self.max_points:]
            self.y2_data = self.y2_data[-self.max_points:]

        if self.x_data and self.y1_data and self.y2_data:
            x = self.x_data[-1]
            y1 = self.y1_data[-1]
            y2 = self.y2_data[-1]

            if self.label1 is None:
                self.label1 = TextItem(color='g', anchor=(0, 1))
                self.plot_widget.addItem(self.label1)
            if self.label2 is None:
                self.label2 = TextItem(color='m', anchor=(1, 1))
                self.plot_widget.addItem(self.label2)

            self.label1.setPos(x, y1)
            self.label1.setText(f"{int(y1)}")
            self.label2.setPos(x, y2)
            self.label2.setText(f"{int(y2)}")

        ymin = min(min(self.y1_data[-20:]), min(self.y2_data[-20:]))
        ymax = max(max(self.y1_data[-20:]), max(self.y2_data[-20:]))
        margin = 50

        self.curve1.setData(self.x_data, self.y1_data)
        self.curve2.setData(self.x_data, self.y2_data)
        self.plot_widget.setXRange(max(0, x - 20), max(0, x + 10))
        self.plot_widget.setYRange(ymin - margin, ymax + margin)

        self.new_y1 = None
        self.new_y2 = None

# ------------------------
# MAIN standalone
# ------------------------
if __name__ == "__main__":
    class LivePlotWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LivePlot Standalone Viewer")
            self.plot_widget = LivePlot()
            self.setCentralWidget(self.plot_widget)

            self.data_timer = QTimer()
            self.data_timer.timeout.connect(self.send_random_data)
            self.data_timer.start(250)

        def send_random_data(self):
            y1 = random.randint(10, 80)   # Simula OCR
            y2 = random.randint(5, 70)    # Simula ICR
            self.plot_widget.add_data_point(y1, y2)

    app = QApplication(sys.argv)
    window = LivePlotWindow()
    window.resize(800, 400)
    window.show()
    sys.exit(app.exec_())
