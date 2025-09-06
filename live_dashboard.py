import collections, time, matplotlib.pyplot as plt
from matplotlib.widgets import Button
from neurosky_connector import NeuroSkyConnector


class LiveDashboard:
    """
    Visualiza atención y meditación en tiempo real con matplotlib,
    además muestra señal y estado de conexión.
    """
    def __init__(self, connector: NeuroSkyConnector, duration: int = 10, fps: int = 20, stop_event=None):
        self.connector = connector
        self.duration = duration
        self.fps = fps
        self.buf_n = duration * fps

        self.stop_event = stop_event

        # Buffers
        self.t_buf = collections.deque(
            [i/self.fps - duration for i in range(self.buf_n)], maxlen=self.buf_n
        )
        self.att_buf = collections.deque([0]*self.buf_n, maxlen=self.buf_n)
        self.med_buf = collections.deque([0]*self.buf_n, maxlen=self.buf_n)

        # Inicializar matplotlib
        plt.ion()

        self.fig = plt.figure(figsize=(10, 8))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 3])
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[2, 0])

        
        self.line_a, = self.ax1.plot([], [], label='Atención', lw=2)
        self.line_m, = self.ax1.plot([], [], label='Meditación', lw=2)
        self.ax1.set_ylim(0, 100)
        self.ax1.set_xlim(-duration, 0)
        self.ax1.legend(loc='upper right')
        self.status_txt = self.ax1.text(
            0.02, 0.95, '', transform=self.ax1.transAxes,
            bbox=dict(boxstyle='round', fc='#ffffff')
        )

        self.quality_txt = self.ax2.text(
            0.5, 0.5, '0%', transform=self.ax2.transAxes,
            fontsize=24, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', pad=0.5, facecolor='red')
        )
        self.ax2.axis('off')

        circle = plt.Circle((0, 0), 1, fill=False)
        self.ax3.add_patch(circle)
        # Ejes horizontales y verticales
        self.ax3.axhline(0, linewidth=1, color='gray')
        self.ax3.axvline(0, linewidth=1, color='gray')
        self.ax3.set_xlim(-1.1, 1.1)
        self.ax3.set_ylim(-1.1, 1.1)
        self.ax3.set_aspect('equal')
        self.ax3.set_xlabel('Valence (Meditation)')
        self.ax3.set_ylabel('Arousal (Attention)')
        self.circ_point, = self.ax3.plot([], [], 'o', markersize=8, color='orange')
        self.ax3.set_title('Circumplex Emocional')
    
                # 1) Conectar el cierre de la ventana al stop_event
        if self.stop_event is not None:
            self.fig.canvas.mpl_connect(
                "close_event",
                lambda evt: self.stop_event.set()
            )

        # 2) (Opcional) Añadir un botón “Stop” en la propia UI
        ax_stop = self.fig.add_axes([0.85, 0.01, 0.1, 0.05])
        btn = Button(ax_stop, "Stop")
        btn.on_clicked(lambda evt: self.stop_event.set())


    def update(self):
        now = time.time()
        # Actualizar buffers de curvas
        self.t_buf.append(now)
        self.att_buf.append(self.connector.attention)
        self.med_buf.append(self.connector.meditation)

        # --- Actualizar curvas atención/meditación ---
        rel = [tt - now for tt in self.t_buf]
        self.line_a.set_data(rel, self.att_buf)
        self.line_m.set_data(rel, self.med_buf)
        self.status_txt.set_text(f"Estado: {self.connector.connection_state}")

        # --- Actualizar señal y estado ---
        pct = int(round(self.connector.signal_quality))
        self.quality_txt.set_text(f"{pct}%")
        if pct >= 90:
            color = 'green'
        elif pct >= 30:
            color = 'orange'
        else:
            color = 'red'
        self.quality_txt.set_bbox(dict(boxstyle='round', pad=0.5, facecolor=color))
        self.status_txt.set_bbox(dict(boxstyle='round', fc=color))

        # --- Actualizar Circumplex Emocional ---
        arousal = (self.connector.attention - 50) / 50
        valence = (self.connector.meditation - 50) / 50
        self.circ_point.set_data([valence], [arousal])

        # Redibujar todo
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(1/self.fps)
