from pymindwave2 import MindWaveMobile2, HeadsetDataEvent, SignalQualityEvent, HeadsetStatusEvent
import sys


class NeuroSkyConnector:
    """
    Encapsula la conexión con el casco MindWaveMobile2.
    Mantiene el estado actualizado (atención, meditación, señal, estado de conexión).
    """
    def __init__(self, n_tries=10, timeout=15):
        self.headset = MindWaveMobile2()
        self.n_tries = n_tries
        self.timeout = timeout

        # Estado interno
        self._last_attention = 0
        self._last_meditation = 0
        self.signal_quality = 0.0
        self.connection_state = "INIT"

        # Suscripciones
        self.headset.on_data(self._on_data)
        self.headset.on_signal_quality_change(self._on_signal)
        self.headset.on_status_change(self._on_status)

    def _on_data(self, ev: HeadsetDataEvent):
        a, m = ev.data.attention, ev.data.meditation
        if a > 0:
            self._last_attention = a
        if m > 0:
            self._last_meditation = m

    def _on_signal(self, ev: SignalQualityEvent):
        self.signal_quality = ev.signal_quality

    def _on_status(self, ev: HeadsetStatusEvent):
        self.connection_state = ev.status.name
        print(f"[HEADSET] Estado de conexión: {self.connection_state}")

    @property
    def attention(self):
        return self._last_attention

    @property
    def meditation(self):
        return self._last_meditation

    def start(self, blocking=False):
        """
        Inicia la conexión con reintentos y timeout.
        """
        success = self.headset.start(n_tries=self.n_tries, timeout=self.timeout)
        if not success:
            print("❌ No se pudo conectar al casco MindWaveMobile2")
            sys.exit(1)
        return success

    def stop(self):
        self.headset.stop()