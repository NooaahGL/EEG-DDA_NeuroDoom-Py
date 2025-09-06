from pymindwave2.session import Session, SessionConfig

import time
import json
import csv
import threading
import signal

from neurosky_connector import NeuroSkyConnector
from udp_sender         import UDPSender
from udp_receiver         import UDPReceiver
from live_dashboard     import LiveDashboard
from user               import UserProfile
from interval_averager          import IntervalAverager

config = SessionConfig(
    user_name="User",
    user_age=0,
    user_gender="F",
    classes=["Preconfigurado", "Heuristica", "ML"],    # Depende el ensayo con usuario
    trials=5,
    
    baseline_duration=10,
    rest_duration=2,
    ready_duration=1,
    cue_duration=1.5,
    motor_duration=4,
    extra_duration=1,
    save_dir="./sessions/",
    capture_blinks=True
)

STOP_EVENT = threading.Event()

def handle_sigint(signum, frame):
    print("\n[Ctrl-C] Señal recibida, cerrando sesión…")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, handle_sigint)

last_ts = None

def main():
    # Configuración
    UDP_IP = '127.0.0.1'
    UDP_PORT  = 5005
    U_IN_PORT = 5006          # Unity→Py
    FPS = 20
    DURATION = 10

    unity_events = []

    def handle_from_unity(pkt: dict):
        global last_ts
        """
        Espera un dict con keys:
            - adaptationMode: str ("Preconfigurado"|"Heuristica"|"ML")
            - horde      : int
            - timestamp  : float  
            - score      : float
        """
        ts = pkt.get("timestamp") / 1000.0

        if ts == last_ts:
            print(f"[Unity→Py] Duplicado ignorado: {ts}")
            return
        last_ts = ts

        stype = str(pkt.get("adaptationMode", ""))
        hnum  = pkt.get("horde")
        score = int(pkt.get("score", 0))

        unity_events.append((stype, hnum, ts, score))
        print(f"[Unity→Py] {stype}, horda {hnum}, score = {score} @ {ts}")

    # Usuario
    profile = UserProfile(name="User", age=99, gender="M")
    
    # Iniciamos conex EEG
    connector = NeuroSkyConnector(n_tries=10, timeout=15)
    connector.start()

    # 3. Creamos el agregador (cada 3s promediará y guardará en usuario). 
    averager = IntervalAverager(profile=profile, interval=3.0)

    # 4. Crear y lanzar la sesión
    session = Session(headset=connector.headset, config=config, lazy_start=False)

    # 5. Iniciamos UDP bidireccional 
    sender = UDPSender(connector, UDP_IP, UDP_PORT, fps=FPS)
    receiver = UDPReceiver(listen_port=5006, on_packet=handle_from_unity)
    receiver.start()

    # 6. Dashboard
    dashboard = LiveDashboard(connector, duration=DURATION, fps=FPS, stop_event=STOP_EVENT)

    try:
        while not STOP_EVENT.is_set():
            # 1) Leer EEG y agregar muestra
            a, m, s = connector.attention, connector.meditation, connector.signal_quality
            averager.add_sample(a, m, s)

            # Actualiza dashboard y envía UDP
            dashboard.update()
            sender.send()

            time.sleep(1.0 / FPS)

    except KeyboardInterrupt:
        print("KeyInterrupt...")

    finally:
        print("⏹  Cerrando recursos…")
        for k, v in profile.summary().items():
            print(f"{k}: {v}")
        print(profile.summary())

        # 1) Cerrar sesión EEG
        try:
            session.stop()
        except KeyError:
            print("Sesión ya desconectada, continúo guardando datos…")
        session.save()
        connector.stop()

        # 2) Cerrar sockets
        receiver.stop()
        sender.close()

        # 3) Volcar los eventos de Unity a CSV en el mismo folder de Session
        save_dir = session._save_dir  
        csv_path = f"{save_dir}/unity_events.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "horde", "timestamp", "score"])  
            writer.writerows(unity_events)                 
        print(f"Unity events guardados en: {csv_path}")

        print("Datos guardados. Programa finalizado.")


if __name__ == '__main__':
    main()
