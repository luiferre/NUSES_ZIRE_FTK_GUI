from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd
import sys, struct

NUM_ASIC = 6  # Numero di ASIC
NUM_CH = 32  # Numero di canali per ASIC
bin_num = 16384  # Numero di bin da visualizzare
bin_iniziale = 0  # Bin iniziale 
DEFAULT_PATH = "/home/gamma/FTK_EQM/DATA/"

DATA_DIM = 16384
SCALE = 16384 // DATA_DIM
# Inizializza istogrammi globali (accumulati su più eventi)
LG_acc = np.zeros((2, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
HG_acc = np.zeros((2, NUM_ASIC, NUM_CH, DATA_DIM), dtype=np.uint32)
bins = np.linspace(0, 16384, DATA_DIM + 1)
timestp_unit = 10e-3

dtCitirocDNI = np.dtype([
    ('Timestamp', '<u8'),
    ('TriggerID', '<u8'),
    ('DAQ1_ID', '<u4'),
    ('DAQ1_TriggerCounts', '<u4'),
    ('DAQ1_Valid', '<u4'),
    ('DAQ1_Rate', '<u4'),
    ('DAQ1_body', '<u4', (192,)),
    ('DAQ1_Lost', '<u8'),
    ('DAQ1_Validated', '<u8'),
    ('DAQ2_ID', '<u4'),
    ('DAQ2_TriggerCounts', '<u4'),
    ('DAQ2_Valid', '<u4'),
    ('DAQ2_Rate', '<u4'),
    ('DAQ2_body', '<u4', (192,)),
    ('DAQ2_Lost', '<u8'),
    ('DAQ2_Validated', '<u8')
])

def result_plot (df, daq):        
    mid_index = df.shape[1] // 2

    data_LG = df.iloc[:, :mid_index].values
    data_HG = df.iloc[:, mid_index:].values

    # Seleziona solo l'intervallo di bin da bin_iniziale a bin_iniziale + bin_num
    HG_selected = data_HG[bin_iniziale:bin_iniziale + bin_num, :]
    LG_selected = data_LG[bin_iniziale:bin_iniziale + bin_num, :]

    
    fig_HG, axs_HG = plt.subplots(NUM_ASIC, 1, figsize=(10, 12), sharex=True)
    fig_HG.suptitle(f"DAQ {daq} - HG")

    for asic in range(NUM_ASIC):
        ax = axs_HG[asic]
        for ch in range(NUM_CH):
            ch_number = asic * NUM_CH + ch
            ax.plot(range(bin_iniziale, bin_iniziale + bin_num), HG_selected[:, ch_number], label=f"CH {ch}", alpha=0.6)
        ax.set_ylabel(f"ASIC {asic + 1}")
        ax.legend(fontsize=6, ncol=4, loc="upper right")

    axs_HG[-1].set_xlabel("Bin ADC")

    
    fig_LG, axs_LG = plt.subplots(NUM_ASIC, 1, figsize=(10, 12), sharex=True)
    fig_LG.suptitle(f"DAQ {daq} - LG")

    for asic in range(NUM_ASIC):
        ax = axs_LG[asic]
        for ch in range(NUM_CH):
            ch_number = asic * NUM_CH + ch
            ax.plot(range(bin_iniziale, bin_iniziale + bin_num), LG_selected[:, ch_number], label=f"CH {ch}", alpha=0.6)
        ax.set_ylabel(f"ASIC {asic + 1}")
        ax.legend(fontsize=6, ncol=4, loc="upper right")

    axs_LG[-1].set_xlabel("Bin ADC")

    plt.show()

def update_histo(data, daq_id):
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

def create_df():
    global LG_acc, HG_acc

    df_LG_1 = pd.DataFrame(LG_acc[0].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
    df_HG_1 = pd.DataFrame(HG_acc[0].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
    df_LG_2 = pd.DataFrame(LG_acc[1].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)
    df_HG_2 = pd.DataFrame(HG_acc[1].reshape(NUM_ASIC * NUM_CH, DATA_DIM).T)

    df = pd.concat([df_LG_1, df_HG_1, df_LG_2, df_HG_2], axis=1)

    return df

def decode_bin(filename):
    
    event_count = 0
    event_list = []

    with open(f"{DEFAULT_PATH}{filename}.bin", "rb") as f:       
        while True:
            header = f.read(6)  # Legge i primi 6 byte

            if len(header) == 0:
                break

            if len(header) < 6:
                print("Header incompleto")
                break
            else:

                id_data, length, pcks_count = struct.unpack("<HHH", header)

                if id_data != 0xA77A:
                    print(f"ID non corrispondente: {hex(id_data)}")
                    return None

                data_pack = f.read(length)
                if len(data_pack) is None:
                    print("Nessun dato")
                    return
                        
                else:
                    event_count += 1
                    if event_count % 1000 == 0:
                        print(f"Event count: {event_count}")
                    
                    timestamp = np.frombuffer(data_pack[0:8], dtype=np.uint64)[0]
                    trig_id = np.frombuffer(data_pack[8:16], dtype=np.uint64)[0]                
                            
                    if len(data_pack) < 817:             
                        data_id, ev_count, valid, data_count = np.frombuffer(data_pack[16:32], dtype=np.uint32)
                        body = np.frombuffer(data_pack[32:800], dtype=np.uint32)

                        data_id1 = ev1_count= valid1= data1_count= body1= lost1= validated1 = 0
                        data_id2= ev2_count= valid2= data2_count= body2= lost2= validated2 = 0

                        if (data_id == 0x00AAAA00):
                            data_id1 = data_id
                            ev1_count = ev_count
                            valid1 = valid
                            data1_count = data_count
                            body1 = body
                            lost1, validated1 = np.frombuffer(data_pack[800:816], dtype=np.uint64)

                        elif (data_id == 0x00BBBB00):
                            data_id2 = data_id
                            ev2_count = ev_count
                            valid2 = valid
                            data2_count = data_count
                            body2 = body
                            lost2, validated2 = np.frombuffer(data_pack[800:816], dtype=np.uint64)
                        
                        else:
                            pass
                    else:
                        data_id1, ev1_count, valid1, data1_count = np.frombuffer(data_pack[16:32], dtype=np.uint32)
                        body1 = np.frombuffer(data_pack[32:800], dtype=np.uint32)
                        lost1, validated1 = np.frombuffer(data_pack[800:816], dtype=np.uint64)

                        data_id2, ev2_count, valid2, data2_count = np.frombuffer(data_pack[816:832], dtype=np.uint32)
                        body2 = np.frombuffer(data_pack[832:1600], dtype=np.uint32)
                        lost2, validated2 = np.frombuffer(data_pack[1600:1616], dtype=np.uint64)


                    event_list.append((
                        timestamp,
                        trig_id,
                        data_id1,
                        ev1_count,
                        valid1,
                        data1_count,
                        body1,
                        lost1,
                        validated1,
                        data_id2,
                        ev2_count,
                        valid2,
                        data2_count,
                        body2,
                        lost2,
                        validated2
                    ))
        
        # Costruisce array numpy strutturato
        data = np.array(event_list, dtype=dtCitirocDNI)

        # Estrazione dei campi in un dizionario, con decodifica Timestamp e body
        events = {
            "Timestamp": (((data["Timestamp"] & 0xFFFFFFFF) << 32) + (data["Timestamp"] >> 32)) * timestp_unit,
            "TriggerID": data["TriggerID"],
            "DAQ1_ID": data["DAQ1_ID"],
            "DAQ1_TriggerCounts": data["DAQ1_TriggerCounts"],
            "DAQ1_Valid": data["DAQ1_Valid"],
            "DAQ1_Rate": data["DAQ1_Rate"],
            "DAQ1_LG": (0x3FFF - (data['DAQ1_body'] >> 14)) & 0x3FFF,
            "DAQ1_HG": (0x3FFF - (data['DAQ1_body'] & 0x3FFF)) & 0x3FFF,
            "DAQ1_Hit": (data['DAQ1_body'] >> 28) & 0x1,
            "DAQ1_Lost": data["DAQ1_Lost"],
            "DAQ1_Validated": data["DAQ1_Validated"],
            "DAQ2_ID": data["DAQ2_ID"],
            "DAQ2_TriggerCounts": data["DAQ2_TriggerCounts"],
            "DAQ2_Valid": data["DAQ2_Valid"],
            "DAQ2_Rate": data["DAQ2_Rate"],
            "DAQ2_LG": (0x3FFF - (data['DAQ2_body'] >> 14)) & 0x3FFF,
            "DAQ2_HG": (0x3FFF - (data['DAQ2_body'] & 0x3FFF)) & 0x3FFF,
            "DAQ2_Hit": (data['DAQ2_body'] >> 28) & 0x1,
            "DAQ2_Lost": data["DAQ2_Lost"],
            "DAQ2_Validated": data["DAQ2_Validated"],
        }

        return events

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python csv_plot.py <filename>")
        sys.exit(1)
    else:
        FILE_NAME = sys.argv[1]            
        events = decode_bin(FILE_NAME)
        plt.hist(events["DAQ2_HG"].T[23], bins=int(16384/4), range =[.5, 16384.5])
        plt.show()