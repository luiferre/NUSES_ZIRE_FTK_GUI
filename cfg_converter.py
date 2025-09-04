import json, struct
import crcmod.predefined

header = 0xAA3455FF
footer = 0xBB6691FF


class Converter:
    def __init__(self, data_path, id):        
        self.datapath = data_path
        self.id = id

    def convert_file(self):
        # Leggi il file JSON
        try:
            with open(self.datapath, 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            return -1, None
        except Exception as e:
            print("Excepion: ", e)
            return -2, e
        id_cfg = self.data["ID_CFG"]

        if id_cfg == "GSSI":
            # Converti i valori in binario per GSSI PAYLOAD
            valori_binari = self.convert_gssi(self.data)
            id_payload = 1926

        elif id_cfg == "FTK":
            # Converti i valori in binario per GSSI PAYLOAD
            valori_binari = self.convert_gssi(self.data)
            id_payload = 1965

        elif id_cfg == "LEM":
            # Converti i valori in binario per LEM PAYLOAD
            valori_binari = self.convert_lem(self.data)
            id_payload = 1907
            
        elif id_cfg == "TERZ":
            # Converti i valori in binario per LEM PAYLOAD
            valori_binari = self.convert_terzina(self.data)
            id_payload = 1993

        # Calcola il checksum CRC
        crc_funzione = crcmod.predefined.mkCrcFun('crc-32')
        checksum_crc = crc_funzione(valori_binari)

        data_output = int.to_bytes(header, 4, byteorder='little')
        data_output +=  int.to_bytes(self.id, 4, byteorder='little')
        data_output +=  int.to_bytes(id_payload, 4, byteorder='little')
        data_output +=  int.to_bytes(len(valori_binari), 4, byteorder='little')
        data_output += valori_binari
        data_output += int.to_bytes(checksum_crc, 4, byteorder='little')
        data_output += int.to_bytes(footer, 4, byteorder='little')
        return data_output, None

    def convert_gssi(self, data):
        valori_binari = bytearray()
        for chiave, valore in data.items():
            if isinstance(valore, dict):
                valori_binari += self.convert_gssi(valore)
            elif isinstance(valore, list):
                valore_convertito = valore[0]
                numero_di_byte = valore[1]
                if isinstance(valore_convertito, str):
                    valore_convertito = valore_convertito.encode()  # Converte la stringa in una sequenza di byte
                else:
                    valore_convertito = int.to_bytes(valore_convertito, length=numero_di_byte, byteorder='little')
                valori_binari += valore_convertito
        return valori_binari
    
    def convert_terzina(self, data):
        valori_binari = bytearray()
        for chiave, valore in data.items():
            if isinstance(valore, dict):
                valori_binari += self.convert_terzina(valore)
            elif isinstance(valore, list):
                valore_convertito = valore[0]
                numero_di_byte = valore[1]
                if isinstance(valore_convertito, str):
                    valore_convertito = valore_convertito.encode()  # Converte la stringa in una sequenza di byte
                else:
                    valore_convertito = int.to_bytes(valore_convertito, length=numero_di_byte, byteorder='little')
                valori_binari += valore_convertito
        return valori_binari
    
    def convert_lem(self, data):
        valori_binari = bytearray()
        for section_name in data:
            registers = data[section_name]
            if section_name != "ID_CFG":
                for reg in registers:
                    value = reg["Value"]
                    valori_binari += struct.pack("<I", value)
        
        return valori_binari


if __name__ == '__main__':
    conv = Converter("D:\\NUSES\\TERZINA_DEBUG\\terzina_cfg.json", 15)
    cfg_bin, e = conv.convert_file()
    with open('D:\\NUSES\\TERZINA_DEBUG\\terzina_converted.bin', 'wb') as file:
        file.write(cfg_bin) 
    