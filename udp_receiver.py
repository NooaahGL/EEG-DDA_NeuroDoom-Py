import socket
import threading
import json

class UDPReceiver:
    """
    Recibe paquetes JSON de Unity vía UDP y los expone en un callback.
    """
    def __init__(self, listen_ip: str = '0.0.0.0', listen_port: int = 5006, on_packet=None):
        self.listen_addr = (listen_ip, listen_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.listen_addr)
        
        # on_packet debe ser una función: def f(pkt: dict): ...
        self.on_packet = on_packet or (lambda pkt: print("Unity->Py:", pkt))
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        print(f"UDPReceiver: escuchando en {self.listen_addr}")

    def _listen_loop(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(2048)
                raw = data.decode('utf-8', errors='replace')
                print(f"RAW UDP de {addr}: '{raw}'")
                pkt = json.loads(raw)
                self.on_packet(pkt)
            except Exception as e:
                print("UDPReceiver error:", e)

    def stop(self):
        self._running = False
        self.sock.close()
