import sys, socket, time, os, struct
import numpy as np
import pandas as pd
import cfg_converter as cfg
import shutil

##############################################################################################
SERVER_IP = "192.168.100.200"
PORT_CONTROL = 3000
PORT_DATA = 2000
SOCK_TIMEOUT = 1
CONFIG_PATH = "/home/gamma/FTK_EQM/CONFIG/ftk_cfg.json"
RESULT_PATH = "/home/gamma/FTK_EQM/DATA"
DATA_DIM = 16384
NUM_ASIC = 6
NUM_CH = 32
SCALE = 16384 // DATA_DIM
DEBUG = False
##############################################################################################################
# Usage: python AUTOMATED.py <time/events/monitor> <seconds/num_evs/cmd> <filename/none>  skip  ##############
##############################################################################################################
# skip to skip the configuration loading 
# cmd for monitor daq_asic_ch_cmd 
# daq = 00 or 01 
# asic = 00-05
# ch = 00-31 
# cmd = reset, fson, fsoff, sshonlg, sshofflg, sshonhg, sshoffhg, paonlg, paofflg, ... 
# ..paonhg, paoffhg, dacon, dacoff, pdonnlg, pdofflg, pdonhg, pdoffhg 
#
# example: python AUTOMATED.py monitor 01_03_15_reset none 
##############################################################################################################

# Inizializza istogrammi globali (accumulati su più eventi)
LG_acc = np.zeros((2, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
HG_acc = np.zeros((2, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
bins = np.linspace(0, 16384, DATA_DIM + 1)

CMD_MAP = {
    "fson":  "fsi",  
    "fsoff": "fso",
    "reset":  "rst",
    "sshonlg":"isl",   
    "sshofflg":"osl",  
    "sshonhg":"ish",   
    "sshoffhg":"osh",  
    "paonlg":"ipl",    
    "paofflg":"opl",   
    "paonhg":"iph",    
    "paoffhg":"oph",   
    "dacon":"idc",     
    "dacoff":"odc",    
    "pdonnlg":"idl",   
    "pdofflg":"odl",  
    "pdonhg":"idh",    
    "pdoffhg":"odh",   
}

class ZIRE_AUTO:
    def __init__(self):
        self.CONTROL = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.CONTROL.settimeout(SOCK_TIMEOUT)

        self.DATA = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.DATA.settimeout(SOCK_TIMEOUT)

    def connect_to_server(self):
        try:
            self.CONTROL.connect((SERVER_IP, PORT_CONTROL))
            self.DATA.connect((SERVER_IP, PORT_DATA))
            print("Connessione stabilita!")
        
        except (socket.error, ConnectionRefusedError):
            print("Errore di connessione")
            return None, None

    def load_cfg(self): 
        file_path = CONFIG_PATH
        try:
            conv = cfg.Converter(file_path, 15)
            cfg_bin, e = conv.convert_file()
            print (f"Ready to sent {len(cfg_bin)} bytes!")
            message = "recvcfgs"
            self.CONTROL.sendall(message.encode())

            aws = self.CONTROL.recv(8).decode()
            if aws == "cfgsdone":
                self.CONTROL.sendall(cfg_bin)
                print("Sent!")
            else:
                print("Error!")

        except FileNotFoundError:   
            return -1, None

    def _recv_exactly(self):
        """ Riceve un pacchetto strutturato: header da 6 byte + body di lunghezza nota. """
        try:
            # Timeout solo per l'header
            self.DATA.settimeout(1)
            received = bytearray()
            while len(received) < 6:
                chunk = self.DATA.recv(6 - len(received))
                if not chunk:
                    print("Connessione chiusa durante la ricezione dell'header")
                    return None
                received.extend(chunk)

            id_data, length, pcks_count = struct.unpack("<HHH", received)

            if id_data != 0xA77A:
                print(f"ID non corrispondente: {hex(id_data)}")
                return None

            # Disabilita il timeout per il corpo del pacchetto
            self.DATA.settimeout(1)
            body = bytearray()
            while len(body) < length:
                chunk = self.DATA.recv(length - len(body))
                if not chunk:
                    print("Connessione chiusa durante la ricezione")
                    return None
                body.extend(chunk)

            if len(body) < 20:
                return None

            with open(f"{RESULT_PATH}/{FILENAME}.bin", "ab") as f:
                f.write(struct.pack("<HHH", id_data, length, pcks_count))
                f.write(body)

            return body

        except socket.timeout:
            print("Timeout: nessun pacchetto disponibile.")
            return None
        
        except Exception as e:
            print(f"Errore durante la ricezione dei dati: {e}")
            return None
    
    def update_histo(self, data, daq_id):
        global LG_acc, HG_acc
        try:
            raw_values = data.reshape(NUM_ASIC, NUM_CH)
            LG = (0x3FFF - (raw_values >> 14)) & 0x3FFF
            HG = (0x3FFF - (raw_values & 0x3FFF)) & 0x3FFF #raw_values[::-1, :] per invertire l'ordine degli asic

            HG_bin = HG // SCALE 
            LG_bin = LG // SCALE  

            daq_idx = daq_id - 1
            np.add.at(LG_acc, (daq_idx, np.arange(NUM_ASIC)[:, None], np.arange(NUM_CH), LG_bin), 1)
            np.add.at(HG_acc, (daq_idx, np.arange(NUM_ASIC)[:, None], np.arange(NUM_CH), HG_bin), 1)
        except Exception as e:
            print(f"Errore nell'aggiornamento dell'istogramma: {e}")

    def save_data_to_csv(self, comment):
        global LG_acc, HG_acc

        df_LG_1 = pd.DataFrame(LG_acc[0].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
        df_HG_1 = pd.DataFrame(HG_acc[0].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
        df_LG_2 = pd.DataFrame(LG_acc[1].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
        df_HG_2 = pd.DataFrame(HG_acc[1].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)

        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        
        df = pd.concat([df_LG_1, df_HG_1, df_LG_2, df_HG_2], axis=1)            
        df.to_csv(f"{RESULT_PATH}/{comment}.csv", index=False, header=False)

def parse_monitor_arg(arg: str):
    """
    Accetta stringa 'daq_asic_ch_cmd' (es. '01_03_15_reset')
    Manda: (cmd3, board, asic, ch)
    """
    try:
        daq_s, asic_s, ch_s, cmd_s = arg.strip().split("_", 3)
        board = int(daq_s)            # il firmware usa board 0/1
        asic  = int(asic_s)            # 0..5
        ch    = int(ch_s)              # 0..31
        if not (0 <= board <= 1 and 0 <= asic <= 5 and 0 <= ch <= 31):
            raise ValueError("Intervalli non validi")

        cmd3 = CMD_MAP.get(cmd_s.lower())
        if not cmd3 or len(cmd3) != 3:
            raise ValueError(f"Comando '{cmd_s}' non mappato a 3 caratteri")
        
        frame = f"m{cmd3}{board}{asic}{ch:02d}"
        print(f"Monitor cmd -> '{frame}'  (cmd3={cmd3}, board={board}, asic={asic}, ch={ch})")

        return frame
    
    except Exception as e:
        raise ValueError(f"Argomento monitor non valido: {arg} ({e})")

############################
########### MAIN ###########
############################

if __name__ == "__main__":
    if len(sys.argv) < 3:
        if DEBUG:
            TYPE = "time"
            COUNTS = 10
            FILENAME = "test_auto"
        else:
            print("Usage: python AUTOMATED.py <time/events> <seconds/num_evs> <filename>")
            sys.exit(1)
    else:
        TYPE = sys.argv[1]
        if TYPE not in ["time", "events", "monitor", "config", "stats"]:
            print("Tipo non valido. Usare 'time', 'events', 'monitor', 'config' or 'stats' .")
            sys.exit(1)
        elif TYPE in ["time", "events"]:
            COUNTS = int(sys.argv[2])
            FILENAME = sys.argv[3]
            
            if os.path.exists(f"{RESULT_PATH}/{FILENAME}.bin"):
                print(f"Il file '{FILENAME}' esiste già. Uscita senza scrivere.")
                sys.exit(1)
            
            shutil.copy(CONFIG_PATH, f"{RESULT_PATH}/{FILENAME}.json")

        skip = False
        if len(sys.argv) > 3:
            if sys.argv[3].lower() == "skip":
                skip = True

    print("Connecting...")
    ZIRE = ZIRE_AUTO()
    ZIRE.connect_to_server()
    
    if not ZIRE.CONTROL or not ZIRE.DATA:
        print("Exiting...")
        sys.exit(1) 
    
    if skip == False:
        print("Sending configuration...")
        if ZIRE.load_cfg() == -1:
            print("Exiting...")
            sys.exit(1)
    
    try:
        if TYPE == "events":
            print(f"Acquisitions for --> [{TYPE} - {COUNTS}] starting...")
            message = "tdatareq"
            ZIRE.CONTROL.sendall(message.encode())
            time.sleep(4)

            event_count = 0   

            while event_count < COUNTS:               
                try:
                    # Ricevi esattamente buffer_size byte senza disallineamenti
                    received = ZIRE._recv_exactly()

                    
                    if received is None:
                        print("Nessun dato ricevuto")
                        continue
                    
                    else:
                        event_count += 1
                        if event_count % 1000 == 0:
                            print(f"Event count: {event_count}")

                    '''
                        data_id = np.frombuffer(received[16:20], dtype=np.uint32)                
                        
                        if len(received) < 817:                
                            data_id1, ev_count, valid, data_count = np.frombuffer(received[16:32], dtype=np.uint32)
                            data_plot = np.frombuffer(received[32:800], dtype=np.uint32)
                            if (data_id == 0x00AAAA00):
                                ZIRE.update_histo(data_plot, 1)
                            elif (data_id == 0x00BBBB00):
                                data_id1, ev_count, valid, data_count = np.frombuffer(received[16:32], dtype=np.uint32)
                                ZIRE.update_histo(data_plot, 2)
                            else:
                                pass
                        else:
                            data1 = np.frombuffer(received[32:800], dtype=np.uint32)
                            data2 = np.frombuffer(received[832:1600], dtype=np.uint32)
                            ZIRE.update_histo(data1, 1)
                            ZIRE.update_histo(data2, 2)
                    '''
                    
                except Exception as e:
                    print(f"Errore : {e}")
                    pass
            
            ZIRE.CONTROL.sendall(b"tdataend")
            time.sleep(1)
            #ZIRE.save_data_to_csv(FILENAME)
            print("Acquisition teminated!")

        elif TYPE == "time":
            print(f"Acquisitions for --> [{TYPE} - {COUNTS}] starting...")
            message = "tdatareq"
            ZIRE.CONTROL.sendall(message.encode())
            time.sleep(4)

            event_count = 0

            start_time = time.time()

            while (time.time() - start_time) < COUNTS:
                try:
                    received = ZIRE._recv_exactly()

                    
                    if received is None:
                        print("Nessun dato ricevuto")
                        continue

                    elif len(received) < 20:
                        continue

                    else:
                        event_count += 1
                        if event_count % 1000 == 0:
                            print(f"Event count: {event_count}")
                    '''    
                        data_id = np.frombuffer(received[16:20], dtype=np.uint32)

                        if len(received) < 817:
                            data_id1, ev_count, valid, data_count = np.frombuffer(received[16:32], dtype=np.uint32)
                            data_plot = np.frombuffer(received[32:800], dtype=np.uint32)

                            if data_id == 0x00AAAA00:
                                ZIRE.update_histo(data_plot, 1)
                            elif data_id == 0x00BBBB00:
                                ZIRE.update_histo(data_plot, 2)
                            else:
                                pass
                        else:
                            data1 = np.frombuffer(received[32:800], dtype=np.uint32)
                            data2 = np.frombuffer(received[832:1600], dtype=np.uint32)
                            ZIRE.update_histo(data1, 1)
                            ZIRE.update_histo(data2, 2)
                    '''

                except Exception as e:
                    print(f"Errore : {e}")
                    pass

            ZIRE.CONTROL.sendall(b"tdataend")
            time.sleep(1)
            #ZIRE.save_data_to_csv(FILENAME)
            print("Acquisition teminated!")

        elif TYPE == "monitor":
            try:
                frame = parse_monitor_arg(sys.argv[2])
                ZIRE.CONTROL.sendall(frame.encode())
                
            except ValueError as e:
                print(e)
                sys.exit(1)

        elif TYPE == "stats":
            try:
                ZIRE.CONTROL.sendall(b"getstats")
                time.sleep(1)
                ZIRE.CONTROL.settimeout(2)
                received = bytearray()
                while len(received) < (2*(192+16)*4):
                    chunk = ZIRE.CONTROL.recv( (2*(192+16)*4) - len(received))
                    if not chunk:
                        print("Connessione chiusa durante la ricezione delle statistiche")
                    received.extend(chunk)
                
                stats = np.frombuffer(received, dtype=np.uint32)

                # Segmentazione
                rate1 = stats[0:192]
                temp1 = stats[192:192+16]
                rate2 = stats[192+16:192+16+192]
                temp2 = stats[192+16+192:]

                print("Rate 1:", rate1)
                print("Temp 1:", temp1)
                print("Rate 2:", rate2)
                print("Temp 2:", temp2)

            except socket.timeout:
                print("Timeout: nessun dato disponibile.")

    except KeyboardInterrupt:
        print("\nInterruzione manuale ricevuta (Ctrl+C)")
