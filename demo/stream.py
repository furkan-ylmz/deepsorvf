import socket
import time

HOST = "127.0.0.1"
PORT = 10110

def stream_nmea(file_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        conn, addr = s.accept()
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                conn.sendall((line + "\n").encode())
                print(f"AIS gönderildi: {line}")
                time.sleep(0.5) 

if __name__ == "__main__":
    stream_nmea("ais_all_sorted2.nmea")
