import socket, json
from neurosky_connector import NeuroSkyConnector

class UDPSender:
    """
    Envía por UDP el estado actual en JSON.
    """
    def __init__(self, connector: NeuroSkyConnector, ip: str, port: int, fps: int = 20):
        self.connector = connector
        self.addr = (ip, port)
        self.interval = 1.0 / fps
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self):
        packet = {
            "attention": self.connector.attention,
            "meditation": self.connector.meditation,
            "signal": self.connector.signal_quality,
            "state": self.connector.connection_state,
        }
        msg = json.dumps(packet).encode('utf-8')
        self.sock.sendto(msg, self.addr)

    def close(self):
        self.sock.close()
