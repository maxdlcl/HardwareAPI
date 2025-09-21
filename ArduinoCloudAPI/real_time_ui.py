# Client ID: 9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW
# Client Secret: CNm0vxo90QHg3VitUgv3o9yNy0EVsqguvteaiVphvR1pfBgQ88xAC2p0cMYkoDYu
# Thing ID: 38fbfa60-5c0b-47e8-aa56-2f5083ffb631

import os
import time
import threading
from collections import deque
from datetime import datetime

# Arduino IoT Cloud API / OAuth2
import iot_api_client as iot
from iot_api_client.configuration import Configuration
from iot_api_client.rest import ApiException

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# GUI / Plot
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración
CLIENT_ID = "9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW"
CLIENT_SECRET = "CNm0vxo90QHg3VitUgv3o9yNy0EVsqguvteaiVphvR1pfBgQ88xAC2p0cMYkoDYu"
THING_ID = "38fbfa60-5c0b-47e8-aa56-2f5083ffb631"

API_HOST = "https://api2.arduino.cc"
TOKEN_URL = f"{API_HOST}/iot/v1/clients/token"
AUDIENCE  = f"{API_HOST}/iot"

POLL_SECONDS = 1           # cada cuánto consultar
WINDOW_SIZE  = 120         

SENSOR_KEYS = ["temp", "hum", "air_q", "light"]  # propiedades esperadas

# Rangos para "gauges"
GAUGE_LIMITS = {
    "temp":  (0, 50),      # °C
    "hum":   (0, 100),     # %
    "air_q": (0, 2000),    # ppm
    "light": (0, 4095),    # ADC 12 bits
}

# Umbrales simples para colorear
THRESHOLDS = {
    "temp":  [(0, 18, "#4caf50"), (18, 28, "#87be48"), (28, 35, "#ffc107"), (35, 100, "#f44336")],
    "hum":   [(0, 30, "#ffc107"), (30, 60, "#4caf50"), (60, 75, "#8bc34a"), (75, 101, "#f44336")],
    "air_q": [(0, 700, "#4caf50"), (700, 1000, "#ffc107"), (1000, 2000, "#f44336")],
    "light": [(0, 200, "#607d8b"), (200, 1200, "#8bc34a"), (1200, 4096, "#ffc107")],
}

def pick_color(sensor, value):
    for lo, hi, color in THRESHOLDS[sensor]:
        if lo <= value < hi:
            return color
    return "#9e9e9e"

def fetch_token():
    oauth_client = BackendApplicationClient(client_id=CLIENT_ID)
    oauth = OAuth2Session(client=oauth_client)
    token = oauth.fetch_token(
        token_url=TOKEN_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        include_client_id=True,
        audience=AUDIENCE,
    )
    return token.get("access_token")

def build_client(access_token):
    cfg = Configuration(host=API_HOST)
    cfg.access_token = access_token
    return iot.ApiClient(cfg)

class IoTReader:
    def __init__(self, client, thing_id):
        self.api = iot.PropertiesV2Api(client)
        self.thing_id = thing_id

    def read_all(self):
        """
        Devuelve un dict con {prop_name: valor_float_o_None} y también
        cualquier propiedad de texto si existe.
        """
        values = {}
        texts  = {}
        resp = self.api.properties_v2_list(self.thing_id)
        for prop in resp:
            name = prop.name
            val  = prop.last_value
            # algunas propiedades pueden ser string (mensajes), otras numéricas
            if isinstance(val, (int, float)):
                values[name] = float(val)
            else:
                # intenta castear si viene como "123"
                try:
                    values[name] = float(val)
                except Exception:
                    texts[name] = val
        return values, texts

class LiveGUI:
    def __init__(self, reader):
        self.reader = reader

        # Buffers de tiempo/series
        self.time_buf = deque(maxlen=WINDOW_SIZE)
        self.buffers = {k: deque(maxlen=WINDOW_SIZE) for k in SENSOR_KEYS}

        # Tk base
        self.root = tk.Tk()
        self.root.title("Arduino IoT Monitor – Gauges + Live Plot")

        # Top frame: gauges
        gauges_frame = ttk.Frame(self.root, padding=12)
        gauges_frame.pack(side=tk.TOP, fill=tk.X)

        self.gauges = {}
        for i, key in enumerate(SENSOR_KEYS):
            f = ttk.Frame(gauges_frame, padding=8)
            f.grid(row=0, column=i, sticky="nsew")
            lbl = ttk.Label(f, text=key.upper(), font=("Segoe UI", 10, "bold"))
            lbl.pack(anchor="w")
            bar = ttk.Progressbar(f, orient="horizontal", length=220, mode="determinate", maximum=GAUGE_LIMITS[key][1]-GAUGE_LIMITS[key][0])
            bar.pack(pady=4)
            val_label = ttk.Label(f, text="—", font=("Segoe UI", 11))
            val_label.pack(anchor="e")
            self.gauges[key] = {"frame": f, "bar": bar, "val_label": val_label}

        # Middle frame: mensajes de texto (propiedades tipo string)
        text_frame = ttk.Frame(self.root, padding=10)
        text_frame.pack(side=tk.TOP, fill=tk.X)
        self.text_box = tk.Text(text_frame, height=4, wrap="word")
        self.text_box.pack(fill=tk.X)
        self.text_box.insert("end", "Mensajes del dispositivo aparecerán aquí...\n")
        self.text_box.config(state="disabled")

        # Bottom frame: plot
        plot_frame = ttk.Frame(self.root, padding=8)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_title("Sensores en tiempo real")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Valor")
        self.lines = {}
        for key in SENSOR_KEYS:
            (line,) = self.ax.plot([], [], label=key)
            self.lines[key] = line
        self.ax.legend(loc="upper left", ncols=4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Estado
        self._stop = False
        self._token_expiry_time = time.time() + 3000
        self._access_token = None
        self._api_client = None

        # Inicia autenticación y loop
        self._ensure_client()
        self._schedule_poll()

        # Manejo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self._stop = True
        self.root.destroy()

    def _ensure_client(self):
        # renueva token si caducó (simple)
        if (self._access_token is None) or (time.time() > self._token_expiry_time - 60):
            try:
                self._access_token = fetch_token()
                self._api_client = build_client(self._access_token)
                self.reader.api = iot.PropertiesV2Api(self._api_client)
                self._token_expiry_time = time.time() + 3300  # ~55min
            except Exception as e:
                self._log_text(f"[ERROR] No se pudo obtener token: {e}")

    def _schedule_poll(self):
        if self._stop:
            return
        # usar after() para mantener el loop en el hilo del GUI
        self.root.after(POLL_SECONDS * 1000, self._poll_once)
        # pero dispara uno inmediato al iniciar
        if not self.time_buf:
            self._poll_once()

    def _poll_once(self):
        if self._stop:
            return
        self._ensure_client()
        try:
            values, texts = self.reader.read_all()
            self._update_buffers(values)
            self._update_gauges(values)
            self._update_plot()
            if texts:
                self._append_texts(texts)
        except ApiException as e:
            self._log_text(f"[API ERROR] {e}")
        except Exception as e:
            self._log_text(f"[ERROR] {e}")

        # reprograma siguiente lectura
        self._schedule_poll()

    def _update_buffers(self, values):
        now = datetime.now().strftime("%H:%M:%S")
        self.time_buf.append(now)
        for key in SENSOR_KEYS:
            v = float(values.get(key)) if key in values and values[key] is not None else float("nan")
            self.buffers[key].append(v)

    def _update_gauges(self, values):
        for key in SENSOR_KEYS:
            low, high = GAUGE_LIMITS[key]
            raw = values.get(key)
            if raw is None:
                self.gauges[key]["val_label"].config(text="NaN")
                self.gauges[key]["bar"]["value"] = 0
                continue
            # recorta a rango y actualiza
            val = max(low, min(high, float(raw)))
            self.gauges[key]["bar"]["maximum"] = high - low
            self.gauges[key]["bar"]["value"] = val - low
            # unidad amigable
            unit = {"temp": "°C", "hum": "%", "air_q": "ppm", "light": ""}.get(key, "")
            self.gauges[key]["val_label"].config(text=f"{val:.0f}{unit}")

            # color por umbral
            color = pick_color(key, val)
            
            self.gauges[key]["frame"].configure(style=f"{key}.TFrame")
            style = ttk.Style()
            style.configure(f"{key}.TFrame", background=color)
            # 
            self.gauges[key]["val_label"].configure(background=color)
            for child in self.gauges[key]["frame"].winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(background=color)

    def _update_plot(self):
        x = range(len(self.time_buf))
        for key in SENSOR_KEYS:
            y = list(self.buffers[key])
            self.lines[key].set_data(x, y)
        # ajusta límites
        self.ax.relim()
        self.ax.autoscale_view()
        # ticks X escasos para legibilidad
        if len(self.time_buf) > 0:
            xticks_idx = list(range(0, len(self.time_buf), max(1, len(self.time_buf)//8)))
            self.ax.set_xticks(xticks_idx)
            self.ax.set_xticklabels([self.time_buf[i] for i in xticks_idx], rotation=0)
        self.canvas.draw_idle()

    def _append_texts(self, texts_dict):
        # imprime mensajes tipo string (como "bad_air", "water", etc.)
        msgs = []
        for k, v in texts_dict.items():
            if v is None or str(v).strip() == "":
                continue
            msgs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {k}: {v}")
        if not msgs:
            return
        self.text_box.config(state="normal")
        for m in msgs:
            self.text_box.insert("end", m + "\n")
        self.text_box.see("end")
        self.text_box.config(state="disabled")

    def _log_text(self, line):
        self.text_box.config(state="normal")
        self.text_box.insert("end", line + "\n")
        self.text_box.see("end")
        self.text_box.config(state="disabled")

    def run(self):
        self.root.mainloop()

def main():
    # cliente inicial
    access_token = fetch_token()
    client = build_client(access_token)
    reader = IoTReader(client, THING_ID)
    gui = LiveGUI(reader)
    gui.run()

if __name__ == "__main__":
    main()

