from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd

NUM_ASIC = 6  # Numero di ASIC
NUM_CH = 32  # Numero di canali per ASIC
bin_num = 16384  # Numero di bin da visualizzare
bin_iniziale = 0  # Bin iniziale 

def result_plot (df):
        
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

df = pd.read_csv("D:\\NUSES\\FTK_DEBUG_v2\\RESULTS\\AUTO\\test_bari.csv", header=None)
mid_index = df.shape[1] // 2
daq_1 = df.iloc[:, :mid_index]
daq_2 = df.iloc[:, mid_index:]
result_plot(daq_1)
result_plot(daq_2)