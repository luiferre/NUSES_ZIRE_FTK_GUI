import sys, socket, struct, os
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import numpy as np
import threading, time
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIM = 16384  
DAQ_NUM = 2
NUM_ASIC = 6
NUM_CH = 32
SCALE = 16384 // DATA_DIM
j_path = "D:\\NUSES\\FTK_DEBUG_v2\\ftk_cfg.json"
RESULT_PATH = "D:\\NUSES\\FTK_DEBUG_v2\\SAVED"
bin_num = 16384  # Numero di bin da visualizzare
bin_iniziale = 0  # Bin iniziale 


# Inizializza istogrammi globali (accumulati su più eventi)
LG_acc = np.zeros((DAQ_NUM, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
HG_acc = np.zeros((DAQ_NUM, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
bins = np.linspace(0, 16384, DATA_DIM + 1)

# Crea la GUI
class Data(QWidget):
    log_signal = pyqtSignal(str)
    icr_ocr_signal = pyqtSignal(int, int)

    def __init__(self, socket, sck_data, main_window=None):          
        super().__init__()
        self.sock = socket
        self.main_window = main_window
        self.sock_data = sck_data
        self.daq_id = 1
        self.mainlat = QVBoxLayout(self)
        self.lg_data = np.zeros(DATA_DIM, dtype=np.uint32)  # Usa un array NumPy per prestazioni migliori
        self.hg_data = np.zeros(DATA_DIM, dtype=np.uint32)  # Usa un array NumPy per prestazioni migliori
        self.sel_asic = 0
        self.sel_ch = 0
        self.old_asic = 0
        self.old_ch = 0
        self.save_data_enabled = False
        self.show_data_enabled = False
        self.event_count = 0
        self.run_id = 0
        self.old_id_comment = "@"
                
        daq = QWidget()
        daq.setAutoFillBackground(True)
        p = daq.palette()
        p.setColor(daq.backgroundRole(), QColor(235, 245, 237))
        daq.setPalette(p)
        self.mainlat.addWidget(daq, 0)
        
        self.mainlayout = QGridLayout()
        daq.setLayout(self.mainlayout)
        label = QLabel(self)
        pixmap = QPixmap("ni-logo-line.png")  # Sostituisci con il percorso dell'immagine
        label.setPixmap(pixmap)

       # Bottoni per avviare e interrompere l'aggiornamento
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_data)
        self.start_button.setFixedSize(100, 40)
        self.start_button.setFont(QFont("Arial", 14))
        self.mainlayout.addWidget(self.start_button, 0, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_data)
        self.stop_button.setFixedSize(100, 40)
        self.stop_button.setFont(QFont("Arial", 14))
        self.stop_button.setEnabled(False)
        self.mainlayout.addWidget(self.stop_button, 0, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.save_button = QPushButton('Save\nNot Enabled')
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setStyleSheet("background-color: lightblue; color: black;")
        self.save_button.setFixedSize(100, 40)
        self.save_button.setFont(QFont("Arial", 10))
        self.save_button.setEnabled(True)
        self.mainlayout.addWidget(self.save_button, 0, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)

        self.show_button = QPushButton('Show Data\nNot Enabled')
        self.show_button.clicked.connect(self.show_data)
        self.show_button.setStyleSheet("background-color: lightblue; color: black;")
        self.show_button.setFixedSize(100, 40)
        self.show_button.setFont(QFont("Arial", 10))
        self.show_button.setEnabled(True)
        self.mainlayout.addWidget(self.show_button, 0, 1, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)

        self.daq1_button = QPushButton('DAQ 1')
        self.daq1_button.clicked.connect(self.daq1_sel)
        self.daq1_button.setFixedSize(100, 40)
        self.daq1_button.setFont(QFont("Arial", 14))
        self.mainlayout.addWidget(self.daq1_button, 1, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        
        self.daq2_button = QPushButton('DAQ 2')
        self.daq2_button.clicked.connect(self.daq2_sel)
        self.daq2_button.setFixedSize(100, 40)
        self.daq2_button.setFont(QFont("Arial", 14))
        self.mainlayout.addWidget(self.daq2_button, 1, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.button_grid = QGridLayout()
        self.mainlayout.addLayout(self.button_grid, 2, 0, 1, 1)
        self.mainlayout.setRowStretch(2, 1)
        self.mainlayout.addWidget(label, 3, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignBottom)
        
        # L'assegnazione nella griglia è all'interno della funzione
        self.button_grid.update()
        self.mainlayout.update()
        self.create_histo()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)  # Collega il timer alla funzione di aggiornamento
        self.timer.start(500)  # Aggiorna ogni 500 ms

        self.channels = []
        for i in range(6):
            button = QPushButton(f"ASIC {i+1}")
            button.setFixedSize(60, 30)
            button.setFont(QFont("Arial", 10))
            button.clicked.connect(lambda _, x=i+1: self.asic_sel(x))
            self.button_grid.addWidget(button, i // 6, i % 6)
            self.channels.append(button)
        
        for i in range(8, 40):
            button = QPushButton(f"CH {i-8}")
            button.setFixedSize(50, 30)
            button.setFont(QFont("Arial", 10))
            button.clicked.connect(lambda _, x=i-7 : self.ch_sel(x))
            self.button_grid.addWidget(button, i // 8 + 1, i % 8)
            self.channels.append(button)

        self.channels[0].setEnabled(False)
        self.channels[6].setEnabled(False)
        self.daq1_sel()
        self.plot_updating = False

    def save_data(self):
        if not self.save_data_enabled:
            self.save_data_enabled = True
            self.save_button.setText("Save\nEnabled")
            self.save_button.setStyleSheet("background-color: lightgreen; color: black;")
            self.log_signal.emit("Salvataggio dati abilitato")
        else:
            self.save_data_enabled = False
            self.save_button.setText("Save\nNot Enabled")
            self.save_button.setStyleSheet("background-color: lightblue; color: black;")
            self.log_signal.emit("Salvataggio dati disabilitato")

    def show_data(self):
        if not self.show_data_enabled:
            self.show_data_enabled = True
            self.show_button.setText("Show Data\nEnabled")
            self.show_button.setStyleSheet("background-color: lightgreen; color: black;")
            self.log_signal.emit("Visione dati abilitata")
        else:
            self.show_data_enabled = False
            self.show_button.setText("Show Data\nNot Enabled")
            self.show_button.setStyleSheet("background-color: lightblue; color: black;")
            self.log_signal.emit("Visione dati disabilitato")

    def ch_sel(self, index):
        for button in self.channels[6:37]:
                button.setEnabled(True)
        
        self.sel_ch = index - 1
        self.log_signal.emit(f"ASIC {self.sel_asic+1}, Channel {self.sel_ch}")
        self.channels[self.sel_ch+6].setEnabled(False)

    def asic_sel(self, index):
        for button in self.channels[0:6]:
                button.setEnabled(True)
        
        self.sel_asic = index - 1
        self.log_signal.emit(f"ASIC {self.sel_asic+1}, Channel {self.sel_ch+1}")
        self.channels[self.sel_asic].setEnabled(False)

    def daq1_sel(self):
        self.daq_id = 1
        self.log_signal.emit("DAQ 1 selezionato")
        self.daq1_button.setEnabled(False)
        self.daq2_button.setEnabled(True)

    def daq2_sel(self):
        self.daq_id = 2
        self.log_signal.emit("DAQ 2 selezionato")
        self.daq2_button.setEnabled(False)
        self.daq1_button.setEnabled(True)
               
    def create_histo(self):
        # Creazione della finestra dei grafici
        self.win = pg.GraphicsLayoutWidget(show=True, title="Histograms")
        self.win.setStyleSheet("background-color: lightblue;")
        self.mainlayout.addWidget(self.win, 0, 2, 4, 1)

        # Configurazione generale
        pg.setConfigOptions(antialias=True)

        # Istogramma LG
        self.plot_lg = self.win.addPlot(title="LG CHANNEL")
        self.plot_lg.setLabel('left', 'COUNTS')
        self.plot_lg.setLabel('bottom', 'ADC CHANNEL SCALED')
        self.curve_lg = self.plot_lg.plot(pen='c')  # Rosso per LG
        self.win.nextRow()  # Passa alla riga successiva per il secondo grafico

        # Istogramma HG
        self.plot_hg = self.win.addPlot(title="HG CHANNEL")
        self.plot_hg.setLabel('left', 'COUNTS')
        self.plot_hg.setLabel('bottom', 'ADC CHANNEL SCALED')
        self.curve_hg = self.plot_hg.plot(pen='y')  # Blu per HG

    def start_data(self):
       if not self.plot_updating:
            if self.save_data_enabled:
                if self.old_id_comment == self.main_window.central_widget.reg_tab.run_id_comment.text():
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Attenzione")
                    msg_box.setText("Il RUN ID COMMENT è uguale al precedente.")
                    msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                    msg_box.setDefaultButton(QMessageBox.Cancel)
        
                    response = msg_box.exec_()

                    if response == QMessageBox.Cancel:
                        return  # L'utente ha premuto Annulla, quindi interrompi

                if self.main_window.central_widget.reg_tab.run_id_comment.text() == "":
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Attenzione")
                    msg_box.setText("Il RUN ID COMMENT è vuoto.")
                    msg_box.setStandardButtons(QMessageBox.Ok)
                    msg_box.exec_()
                    return
            
            self.plot_updating = True
            
            message = "tdatareq"
            self.sock.sendall(message.encode())
            self.start_button.setEnabled(False)
            for button in self.channels:
                button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.show_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.daq2_button.setEnabled(False)
            self.daq1_button.setEnabled(False)
            self.reset_histo() 
            time.sleep(2)           
            self.run_id += 1
            self.event_count = 0
            thread = threading.Thread(target=self.run, daemon=True)
            thread.start()

    def stop_data(self):
        if self.plot_updating:
            self.plot_updating = False
            self.start_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.show_button.setEnabled(True)
            
            for button in self.channels:
                button.setEnabled(True)
            self.channels[self.sel_asic].setEnabled(False)
            self.channels[self.sel_ch+6].setEnabled(False)
            if self.daq_id == 1:
                self.daq1_button.setEnabled(False)
                self.daq2_button.setEnabled(True)
            else:
                self.daq2_button.setEnabled(False)
                self.daq1_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            message = "tdataend"
            self.sock.sendall(message.encode())
            time.sleep(0.5)
            self.main_window.central_widget.general_tab.reset_plot()
            self.save_data_to_csv()

    def save_data_to_csv(self):
        global LG_acc, HG_acc

        df_LG_1 = pd.DataFrame(LG_acc[0].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
        df_HG_1 = pd.DataFrame(HG_acc[0].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
        df_LG_2 = pd.DataFrame(LG_acc[1].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
        df_HG_2 = pd.DataFrame(HG_acc[1].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)

        if self.save_data_enabled:
            if not os.path.exists(RESULT_PATH):
                os.makedirs(RESULT_PATH)
            
            run_id_comment = self.main_window.central_widget.reg_tab.run_id_comment.text()
            if not os.path.exists(f"{RESULT_PATH}\\{run_id_comment}"):
                os.makedirs(f"{RESULT_PATH}\\{run_id_comment}")
            
            df = pd.concat([df_LG_1, df_HG_1, df_LG_2, df_HG_2], axis=1)            
            df.to_csv(f"{RESULT_PATH}\\{run_id_comment}\\data.csv", index=False, header=False)
            shutil.copy(j_path, f"{RESULT_PATH}\\{run_id_comment}\\config.json")
            self.old_id_comment = run_id_comment


        if self.show_data_enabled:
            if self.daq_id == 1:
                df_plot = pd.concat([df_LG_1, df_HG_1], axis=1)
            else:
                df_plot = pd.concat([df_LG_2, df_HG_2], axis=1)

            self.result_plot(df_plot)

        LG_acc = np.zeros((DAQ_NUM, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
        HG_acc = np.zeros((DAQ_NUM, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)

    def run(self):
        while self.plot_updating:
            try:
                count_rates, data = self._recv_exactly()
                if data is None or data.size == 0:
                    #self.log_signal.emit("Nessun dato ricevuto")
                    continue

                else:
                    self.event_count += 1
                    if self.event_count % 1000 == 0:
                        self.log_signal.emit(f"Event count: {self.event_count}")

                    ocr = (count_rates >> 16) & 0xFFFF
                    icr = count_rates & 0xFFFF
                    self.icr_ocr_signal.emit(ocr, icr)

                    # istogramma corrente (preview)
                    idx = self.sel_asic * 32 + self.sel_ch
                    raw = int(data[idx])
                    HG = 0x3FFF - (raw & 0x3FFF)
                    LG = 0x3FFF - ((raw >> 14) & 0x3FFF)

                    scale = 16384 // DATA_DIM
                    hg_bin = min(HG // scale, DATA_DIM - 1)
                    lg_bin = min(LG // scale, DATA_DIM - 1)

                    self.hg_data[hg_bin] += 1
                    self.lg_data[lg_bin] += 1
                    
            except Exception as e:
                self.log_signal.emit(f"Errore run(): {e}")

    def update_histo(self, data, daq_id):
        global LG_acc, HG_acc
        raw_values = data.reshape(NUM_ASIC, NUM_CH)
        LG = 0x3FFF - ((raw_values >> 14) & 0x3FFF)
        HG = 0x3FFF - (raw_values & 0x3FFF)
        HG_bin = HG // SCALE
        LG_bin = LG // SCALE

        daq_idx = daq_id - 1
        np.add.at(LG_acc, (daq_idx, np.arange(NUM_ASIC)[:, None], np.arange(NUM_CH), LG_bin), 1)
        np.add.at(HG_acc, (daq_idx, np.arange(NUM_ASIC)[:, None], np.arange(NUM_CH), HG_bin), 1)

    def update_plot(self):
        """ Aggiorna il grafico nel thread principale usando QTimer """
        self.curve_hg.setData(self.hg_data)
        self.curve_lg.setData(self.lg_data)

    def result_plot(self, df):
        mid_index = df.shape[1] // 2

        data_LG = df.iloc[:, :mid_index].values
        data_HG = df.iloc[:, mid_index:].values

        # Seleziona solo l'intervallo di bin da bin_iniziale a bin_iniziale + bin_num
        HG_selected = data_HG[bin_iniziale:bin_iniziale + bin_num, :]
        LG_selected = data_LG[bin_iniziale:bin_iniziale + bin_num, :]

        
        fig_HG, axs_HG = plt.subplots(NUM_ASIC, 1, figsize=(10, 12), sharex=True)
        fig_HG.suptitle("HG")

        for asic in range(NUM_ASIC):
            ax = axs_HG[asic]
            for ch in range(NUM_CH):
                ch_number = asic * NUM_CH + ch
                ax.plot(range(bin_iniziale, bin_iniziale + bin_num), HG_selected[:, ch_number], label=f"CH {ch}", alpha=0.6)
            ax.set_ylabel(f"ASIC {asic + 1}")
            ax.legend(fontsize=6, ncol=4, loc="upper right")

        axs_HG[-1].set_xlabel("Bin ADC")

        
        fig_LG, axs_LG = plt.subplots(NUM_ASIC, 1, figsize=(10, 12), sharex=True)
        fig_LG.suptitle("LG")

        for asic in range(NUM_ASIC):
            ax = axs_LG[asic]
            for ch in range(NUM_CH):
                ch_number = asic * NUM_CH + ch
                ax.plot(range(bin_iniziale, bin_iniziale + bin_num), LG_selected[:, ch_number], label=f"CH {ch}", alpha=0.6)
            ax.set_ylabel(f"ASIC {asic + 1}")
            ax.legend(fontsize=6, ncol=4, loc="upper right")

        axs_LG[-1].set_xlabel("Bin ADC")

        # 🔹 Mostra le due figure interattivamente
        plt.show()

    def reset_histo(self):
        self.hg_data.fill(0)
        self.lg_data.fill(0)
        self.curve_hg.setData(self.hg_data)
        self.curve_lg.setData(self.lg_data)

    def _recv_exactly(self):
        """Header: 6B (id,len,cnt) little-endian; body: length bytes."""
        try:
            # 1) leggi header
            self.sock_data.settimeout(3)
            received = bytearray()
            while len(received) < 6:
                chunk = self.sock_data.recv(6 - len(received))
                if not chunk:
                    print("Connessione chiusa durante la ricezione dell'header")
                    return None
                received.extend(chunk)

            id_data, length, pcks_count = struct.unpack("<HHH", received)
            
            if id_data != 0xA77A:
                self.log_signal.emit(f"ID non corrispondente: {hex(id_data)}")
                return None, None

            # 2) leggi body esattamente
            self.sock_data.settimeout(3)
            received = bytearray()
            while len(received) < length:
                chunk = self.sock_data.recv(length - len(received))
                if not chunk:
                    self.log_signal.emit("Connessione chiusa durante la ricezione")
                    return None, None
                received.extend(chunk)

            # 3) parse pack 1 (controllo lunghezza minima)
            if length == 16:
                return None, None
            
            if length < 800:
                self.log_signal.emit(f"Lunghezza inattesa: {length}")
                return None, None

            # scalari, non array (evita ambiguità booleane)
            daq_id_1      = int.from_bytes(received[16:20],  "little", signed=False)
            count_rates_1 = int.from_bytes(received[28:32],  "little", signed=False)
            # 768B -> 192 uint32 little-endian
            data_1        = np.frombuffer(received, dtype="<u4", offset=32, count=192).copy()

            # 4) parse pack 2 se presente
            daq_id_2 = count_rates_2 = None
            data_2 = None
            if length >= 1616:
                daq_id_2      = int.from_bytes(received[816:820], "little", signed=False)
                count_rates_2 = int.from_bytes(received[828:832], "little", signed=False)
                data_2        = np.frombuffer(received, dtype="<u4", offset=832, count=192).copy()

            # 5) selezione DAQ (confronti su SCALARI)
            if self.daq_id == 1 and daq_id_1 == 0x00AAAA00:
                self.update_histo(data_1, 1)
                return count_rates_1, data_1

            if self.daq_id == 2 and daq_id_1 == 0x00BBBB00:
                self.update_histo(data_1, 2)
                return count_rates_1, data_1

            if self.daq_id == 2 and daq_id_2 == 0x00BBBB00:
                # se arrivano entrambi, aggiorna anche DAQ1
                self.update_histo(data_1, 1)
                self.update_histo(data_2, 2)
                return count_rates_2, data_2

            return None, None

        except socket.timeout:
            self.log_signal.emit("Timeout: nessun pacchetto disponibile.")
            return None, None
        
        except Exception as e:
            self.log_signal.emit(f"Errore durante la ricezione dei dati: {e}")
            return None, None


        

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    
    server_address = ('192.168.102.159', 3000)
    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_cmd.connect(server_address)
    sock_cmd.settimeout(1)

    server_address = ('192.168.102.159', 2000)
    sock_data = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_data.connect(server_address)
    sock_data.settimeout(1)

    central_widget = Data(sock_cmd, sock_data, None)
    window.setCentralWidget(central_widget)
    window.show()
    sys.exit(app.exec_())