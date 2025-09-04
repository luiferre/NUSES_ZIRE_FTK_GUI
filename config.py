import sys, socket, struct, threading, os, platform
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QFileDialog 
import cfg_converter as cfg
import subprocess


# Carica i dati dal file JSON
j_path = "D:\\NUSES\\FTK_DEBUG_v2\\ftk_cfg.json"
SERVER_IP = "192.168.102.159"

# Crea la GUI
class Reg(QWidget):
    def __init__(self, socket, main_window=None):
        super().__init__()
        self.sock = socket
        self.main_window=main_window
        self.mainlayout = QVBoxLayout(self)

        #TAB CONFIG
        tab_config = QWidget()
        tab_config.setAutoFillBackground(True)
        p = tab_config.palette()
        p.setColor(tab_config.backgroundRole(), QColor(216, 245, 203))
        tab_config.setPalette(p)
        self.mainlayout.addWidget(tab_config, 0)

        # BUTTONS
        layout_wrapper = QVBoxLayout()
        buttons_widget = QWidget()
        buttons_layout = QGridLayout(buttons_widget)

        send_cfg = QPushButton("OPEN CONFIG")
        send_cfg.setFixedSize(100, 30)
        send_cfg.clicked.connect(self.open_cfg)
        buttons_layout.addWidget(send_cfg, 0, 0)

        load_data = QPushButton("LOAD CONFIG")
        load_data.setFixedSize(100, 30)
        load_data.clicked.connect(self.load_cfg)
        buttons_layout.addWidget(load_data, 0, 1)
                
        self.run_id_comment = QLineEdit()
        self.run_id_comment.setFixedSize(120, 20)
        self.run_id_comment.setText("TEST_0") 
        buttons_layout.addWidget(self.run_id_comment, 1, 1)
        label = QLabel("RUN ID COMMENT")
        buttons_layout.addWidget(label, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)

        dac_offset = QPushButton("SET DAC OFFSET")
        dac_offset.setFixedSize(100, 30)
        dac_offset.clicked.connect(self.set_dac_offset)
        #buttons_layout.addWidget(dac_offset, 2, 0)
        self.dac_val = QLineEdit()
        self.dac_val.setFixedSize(60, 20)
        self.dac_val.setText("0") 
        #buttons_layout.addWidget(self.dac_val, 2, 1)
        label = QLabel("0-4095 su 2.6V / 160 ~ 100mV")
        #buttons_layout.addWidget(label, 2, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        '''
        self.reg_add = QLineEdit()
        self.reg_add.setFixedSize(60, 20)
        self.reg_add.setText("0") 
        buttons_layout.addWidget(self.reg_add, 2, 2)
        nome_label = QLabel("ADDRESS")
        buttons_layout.addWidget(nome_label, 2, 1, alignment=Qt.AlignmentFlag.AlignJustify)

        self.reg_val = QLineEdit()
        self.reg_val.setFixedSize(60, 20)
        self.reg_val.setText("0") 
        buttons_layout.addWidget(self.reg_val, 2, 4)
        label = QLabel("VALUE")
        buttons_layout.addWidget(label, 2, 3, alignment=Qt.AlignmentFlag.AlignJustify)

        send_cfg = QPushButton("W")
        send_cfg.setFixedSize(50, 30)
        send_cfg.clicked.connect(self.write_reg)
        buttons_layout.addWidget(send_cfg, 2, 5)

        send_cfg = QPushButton("R")
        send_cfg.setFixedSize(50, 30)
        send_cfg.clicked.connect(self.read_reg)
        buttons_layout.addWidget(send_cfg, 2, 6)
        '''

        buttons_layout.setColumnStretch(10, 5)
        #buttons_layout.setRowStretch(3,1)
        
        # Imposta il layout per buttons_widget
        buttons_widget.setLayout(buttons_layout)

        # Aggiungi il widget dei bottoni nel layout_wrapper
        layout_wrapper.addWidget(buttons_widget)

        # Imposta il layout nel frame della categoria
        tab_config.setLayout(layout_wrapper)
        
        # Set defaults
        message = "setdeflt"
        self.sock.sendall(message.encode())

    
    def set_dac_offset(self):       
        message = "dq" + f"{int(self.dac_val.text()):06X}"
        self.sock.sendall(message.encode())
       

    def open_cfg(self):
        try:
            if platform.system() == "Windows":
                os.startfile(j_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", j_path])
            else:  # Linux/Unix
                subprocess.run(["xdg-open", j_path])
        except Exception as e:
            self.main_window.log(f"Errore durante l'apertura del file: {e}")

    def load_cfg(self):
        '''
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(None, "Seleziona un file JSON", "", "JSON Files (*.json);;All Files (*)", options=options)   
        if not file_path:
            return
        
        '''
        file_path = j_path
        try:
            conv = cfg.Converter(file_path, 15)
            cfg_bin, e = conv.convert_file()
            self.main_window.log(f"Ready to sent {len(cfg_bin)} bytes!")
            message = "recvcfgs"
            self.sock.sendall(message.encode())

            aws = self.sock.recv(8).decode()
            if aws == "cfgsdone":
                self.sock.sendall(cfg_bin)
                self.main_window.log("Sent!")
            else:
                self.main_window.log("Error!")

        except FileNotFoundError:   
            return -1, None

            
# Classe principale che estende QMainWindow per gestire il closeEvent
class MainWindow(QMainWindow):
    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        self.setGeometry(100, 100, 600, 400)

        # Imposta il widget centrale
        self.central_widget = Reg(sock)
        self.setCentralWidget(self.central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    server_address = (SERVER_IP, 3000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    # Usa la nuova classe MainWindow
    window = MainWindow(sock)
    window.show()
    
    sys.exit(app.exec_())