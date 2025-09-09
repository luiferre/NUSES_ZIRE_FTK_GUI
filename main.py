import sys, socket
from PyQt5.QtGui import QIcon, QTextCursor, QFont
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QGridLayout, QWidget, QAction, QMessageBox, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
import config, data, icr_ocr

SERVER_IP = "192.168.100.200"
PORT_CONTROL = 3000
PORT_DATA = 2000
SOCK_TIMEOUT = 0.1

# 🔹 Classe per la gestione automatica della riconnessione dei socket
class SocketManager(QThread):
    socket_reconnected = pyqtSignal(object, object) 

    def __init__(self, parent=None):
        super().__init__()
        self.running = True
        self.sock_cmd = None
        self.sock_data = None
        self.main_window = parent

    def connect_to_server(self):
        """Crea e restituisce una nuova connessione socket"""
        try:
            sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_cmd.settimeout(SOCK_TIMEOUT)
            sock_cmd.connect((SERVER_IP, PORT_CONTROL))

            sock_data = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_data.settimeout(SOCK_TIMEOUT)
            sock_data.connect((SERVER_IP, PORT_DATA))

            self.main_window.log("Connessione ristabilita!")
            return sock_cmd, sock_data

        except (socket.error, ConnectionRefusedError):
            self.main_window.log("Errore di connessione, tentativo di riconnessione...")
            return None, None

    def run(self):
        """Tenta la connessione al server"""
        self.sock_cmd, self.sock_data = self.connect_to_server()
        self.socket_reconnected.emit(self.sock_cmd, self.sock_data)

# 🔹 Classe principale con i widget
class CentralWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.main_window = parent
        
        # Creazione di un QVBoxLayout
        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)

        # Creazione di un widget
        self.tab_layout = QTabWidget()


        # Inizializzazione delle schede (i socket verranno aggiornati successivamente)
        self.data_tab1 = None
        self.data_tab2 = None
        self.reg_tab = None
        self.general_tab = None

        self.mainlayout.setRowStretch(1, 1)
        self.mainlayout.setColumnStretch(1, 1)

    def update_sockets(self, new_sock_cmd, new_sock_data):

        if self.data_tab1:
            self.data_tab1.sock = new_sock_cmd
            self.data_tab1.sock_data = new_sock_data
        if self.reg_tab:
            self.reg_tab.sock = new_sock_cmd

        # crea i widget solo se mancanti
        if not self.data_tab1:
            self.data_tab1 = data.Data(new_sock_cmd, new_sock_data, main_window=self.main_window)
            self.tab_layout.addTab(self.data_tab1, "DAQ")
            self.mainlayout.addWidget(self.tab_layout, 1, 0, 2, 3)

        if not self.reg_tab:
            self.reg_tab = config.Reg(new_sock_cmd, main_window=self.main_window)
            self.mainlayout.addWidget(self.reg_tab, 0, 0, 1, 1)

        if not self.general_tab:
            self.general_tab = icr_ocr.LivePlot(update_interval_ms=500, parent=self.main_window)
            self.mainlayout.addWidget(self.general_tab, 0, 1, 1, 1)

        self.data_tab1.log_signal.connect(self.main_window.log)
        self.data_tab1.icr_ocr_signal.connect(self.general_tab.add_data_point)

def on_file_clicked(socket_manager):
    # Connetti il segnale di riconnessione all'update dei socket
    socket_manager.start()

# 🔹 Classe per la finestra principale con la menu bar
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('FTK GUI')

        # Creazione del widget centrale
        self.central_widget = CentralWidget(self)
        self.setCentralWidget(self.central_widget)

        # Terminale condiviso
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFixedHeight(120)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #FAFEFF;       /* sfondo nero */
                color: #0C96C0;                  /* testo verde */
                font-family: Consolas, monospace, bold;
                font-size: 12pt;
                border: 2px solid #35A5C7;       /* bordo verde */
                border-radius: 4px;              /* angoli leggermente arrotondati */
                padding: 5px;                    /* spazio interno */
            }
        """)

        # Aggiungi terminale alla fine del layout principale
        self.central_widget.mainlayout.addWidget(self.terminal, 3, 0, 1, 3)

        # Creazione della barra dei menu
        menubar = self.menuBar()

        # Creazione del Socket Manager
        self.socket_manager = SocketManager(self)

        # Azione "File" (cliccabile direttamente)
        file_action = QAction("Riconnetti", self)
        file_action.triggered.connect(lambda: on_file_clicked(self.socket_manager))
        menubar.addAction(file_action)
        
        # Connetti il segnale di riconnessione all'update dei socket
        self.socket_manager.socket_reconnected.connect(self.central_widget.update_sockets)
        
        # Avvia il tentativo di riconnessione
        self.socket_manager.start()

    def log(self, message):
        self.terminal.append(message)
        self.terminal.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        try:
            reply = QMessageBox.question(self, 'Conferma Uscita',
                                         "Sei sicuro di voler chiudere l'applicazione?",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)

            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
                
        except Exception as e:
            self.log(f"Errore durante la chiusura dell'applicazione: {e}")
            event.ignore()

# 🔹 Avvio dell'applicazione
if __name__ == "__main__":
    app = pg.mkQApp("FTK GUI")
    app.setFont(QFont("DejaVu Sans", 9))  # o "Arial", 9
    
    screen = app.primaryScreen()
    screen_geometry = screen.availableGeometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()

    window = MainWindow()
    icon = QIcon("icon.png")
    window.setWindowIcon(icon)
    window.resize(int(screen_width * 0.8), int(screen_height * 0.8))
    window.move(screen_geometry.center() - window.frameGeometry().center())
    
    window.show()
    sys.exit(app.exec_())
