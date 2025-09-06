# NeuroDoom-Python

Código en Python para el prototipo **NeuroDoom**, desarrollado en el marco del Trabajo Fin de Máster:  
*Ajuste dinámico de la dificultad en videojuegos basado en estados emocionales del jugador mediante dispositivos BCI*.

Este repositorio implementa la capa de adquisición EEG, comunicación con el prototipo en Unity y análisis estadístico de las hipótesis H1–H3.  

---

## ✨ Funcionalidades principales

- **Adquisición EEG** con el dispositivo *NeuroSky MindWave Mobile 2*.  
- **Plan de contingencia de conexión**: reintentos automáticos, *timeouts*, abortado controlado y preservación de la consistencia de los datos.  
- **Comunicación UDP** bidireccional con Unity para sincronizar hordas, estados y puntuaciones.  
- **Preprocesado de señales**: medias de atención/meditación, filtrado de ruido y agregación por intervalos.  
- **Pipeline de análisis** para validar hipótesis:
  - H1: Validez de la detección (predicciones vs. estados reales).  
  - H2: Permanencia en *Flow*.  
  - H3: Calidad de la experiencia (encuestas subjetivas).  
- **Generación de resultados**: tablas y figuras incluidas en la memoria del TFM.

---

## ✨ Clonar el repositorio
git clone https://github.com/nooaahGL/eeg-dda_neurodoom-py.git
cd neurodoom-python

---

## 📜 Licencia
Este proyecto se distribuye bajo licencia **MIT**.
