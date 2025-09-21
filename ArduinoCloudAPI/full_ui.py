import time, threading, math, tkinter as tk
from tkinter import ttk
from collections import deque

# Arduino Cloud SDK / OAuth
import iot_api_client as iot
from iot_api_client.configuration import Configuration
from iot_api_client.rest import ApiException
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# ------- Matplotlib embed -------
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Credenciales
CLIENT_ID = "9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW"
CLIENT_SECRET = "CNm0vxo90QHg3VitUgv3o9yNy0EVsqguvteaiVphvR1pfBgQ88xAC2p0cMYkoDYu"
THING_ID = "38fbfa60-5c0b-47e8-aa56-2f5083ffb631"

# Credencales como variables de entorno
# import os
# CLIENT_ID = os.getenv("ARDUINO_CLIENT_ID")
# CLIENT_SECRET = os.getenv("ARDUINO_CLIENT_SECRET")
# THING_ID = os.getenv("ARDUINO_THING_ID")

# Config API
API_HOST = "https://api2.arduino.cc"            # sin "/iot" (error en documentación)          
TOKEN_URL = f"{API_HOST}/iot/v1/clients/token"
AUDIENCE  = f"{API_HOST}/iot"

# Sensores y rangos de los gauges
GAUGES = [
    {"prop": "temp",  "title": "Temp Gauge",     "min": 0,   "max": 100,  "units": "°C"},
    {"prop": "hum",   "title": "Hum Percentage", "min": 0,   "max": 100,  "units": "%"},
    {"prop": "air_q", "title": "Air Q Gauge",    "min": 0,   "max": 1000, "units": ""},
    {"prop": "light", "title": "Light Gauge",    "min": 0,   "max": 5000, "units": "lux"},
]
SENSOR_NAMES = [g["prop"] for g in GAUGES]

# Ventana deslizante para la gráfica
WINDOW_SIZE = 120    
POLL_SECS   = 1 

# Helper Token/API
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
    return token["access_token"]

def build_api(access_token: str):
    cfg = Configuration(host=API_HOST)
    cfg.access_token = access_token
    return iot.ApiClient(cfg)

# Widget: Gauge UI
class SemiCircularGauge(ttk.Frame):
    """
    Gauge semicircular (canvas). Muestra valor dentro de [min,max]
    con arco gris de fondo y arco de valor en color.
    """
    def __init__(self, master, title, min_val, max_val, units="", width=320, height=150):
        super().__init__(master)
        self.min = float(min_val)
        self.max = float(max_val)
        self.units = units
        self.w = width
        self.h = height
        self.pad = 14

        self.title_label = ttk.Label(self, text=title)
        self.title_label.pack(anchor="w", padx=4, pady=(0,4))

        self.canvas = tk.Canvas(self, width=self.w, height=self.h, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=False)

        self.value_label = ttk.Label(self, text="–", font=("TkDefaultFont", 18, "bold"))
        self.value_label.place(x=self.w/2, y=self.h/2, anchor="center")

        self._draw_static()
        self.update_value(None)

    def _draw_static(self):
        self.canvas.delete("all")
        margin = self.pad
        # Bounding box del círculo completo (para dibujar medio arco)
        self.arc_bbox = (margin, margin, self.w - margin, self.h*2 - margin)

        # fondo gris
        self.canvas.create_arc(
            *self.arc_bbox, start=180, extent=180,
            style=tk.ARC, width=18, outline="#e9ecef"
        )
        # ticks extremos
        self.canvas.create_text(self.pad + 6, self.h - 10, text=str(int(self.min)), fill="#6c757d", anchor="w")
        self.canvas.create_text(self.w - self.pad - 6, self.h - 10, text=str(int(self.max)), fill="#6c757d", anchor="e")
        # arco de valor (se actualiza)
        self.value_arc = self.canvas.create_arc(
            *self.arc_bbox, start=180, extent=0,
            style=tk.ARC, width=18, outline="#2aa198"
        )

    def _value_to_extent(self, val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        v = max(self.min, min(self.max, float(val)))
        frac = (v - self.min) / (self.max - self.min + 1e-9)
        return 180.0 * frac

    def update_value(self, val):
        extent = self._value_to_extent(val)
        self.canvas.itemconfigure(self.value_arc, extent=extent)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            txt = "NaN"
        else:
            try:
                num = float(val)
                txt = f"{num:.3f}".rstrip("0").rstrip(".")
            except:
                txt = str(val)
        self.value_label.config(text=f"{txt} {self.units}".strip())

# App principal
class RealTimeGUI:
    def __init__(self, root):
        self.root = root
        root.title("Ambient Monitor (Gauges + Plot) – Arduino Cloud")
        root.configure(bg="white")

        # Layout: fila superior gauges, fila inferior plot
        top = ttk.Frame(root)
        top.pack(fill="x", padx=12, pady=(12, 6))
        bottom = ttk.Frame(root)
        bottom.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Gauges
        self.gauges = {}
        for i, gcfg in enumerate(GAUGES):
            g = SemiCircularGauge(
                top, gcfg["title"], gcfg["min"], gcfg["max"], gcfg["units"],
                width=320, height=150
            )
            g.grid(row=0, column=i, padx=8, pady=4, sticky="nsew")
            top.columnconfigure(i, weight=1)
            self.gauges[gcfg["prop"]] = g

        # Plot (matplotlib)
        self.fig = plt.figure(figsize=(9,4))
        self.ax = self.fig.add_subplot(111)
        self.lines = {}
        self.data = {k: deque(maxlen=WINDOW_SIZE) for k in SENSOR_NAMES}
        self.x = deque(maxlen=WINDOW_SIZE)

        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax.set_title("Time Series")
        for k in SENSOR_NAMES:
            (line,) = self.ax.plot([], [], label=k)
            self.lines[k] = line
        self.ax.legend(loc="upper left")
        self.ax.set_xlabel("t (ticks)")
        self.ax.set_ylabel("valor")
        self.ax.grid(True)

        # Status bar
        self.status = ttk.Label(root, text="Inicializando…")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # API Arduino Cloud
        if not all([CLIENT_ID, CLIENT_SECRET, THING_ID]):
            raise RuntimeError("Faltan CLIENT_ID / CLIENT_SECRET / THING_ID (rellena arriba).")
        self.access_token = fetch_token()
        self.api = build_api(self.access_token)
        self.props_api = iot.PropertiesV2Api(self.api)

        # Loop
        self.tick = 0
        self._stop = False
        threading.Thread(target=self.loop, daemon=True).start()
        self.update_plot()

    def loop(self):
        while not self._stop:
            try:
                props = self.props_api.properties_v2_list(THING_ID)
                current = {p.name: p.last_value for p in props}

                self.tick += 1
                self.x.append(self.tick)

                # Actualiza buffers + gauges
                for gcfg in GAUGES:
                    name = gcfg["prop"]
                    raw = current.get(name, "")
                    if raw == "" or raw is None:
                        val = float("nan")
                    else:
                        try:
                            val = float(raw)
                        except:
                            val = float("nan")

                    # buffers serie
                    if name in self.data:
                        self.data[name].append(val)

                    # gauge UI (thread-safe via 'after')
                    self.root.after(0, self.gauges[name].update_value, val)

                self.status.config(text=f"OK | tick={self.tick}")

            except ApiException as e:
                if e.status == 401:
                    # refresca token
                    try:
                        self.access_token = fetch_token()
                        self.api = build_api(self.access_token)
                        self.props_api = iot.PropertiesV2Api(self.api)
                        self.status.config(text="Token refrescado (401)")
                    except Exception as ee:
                        self.status.config(text=f"Auth error: {ee}")
                else:
                    self.status.config(text=f"API ERROR {e.status}")
            except Exception as e:
                self.status.config(text=f"ERR: {e}")

            time.sleep(POLL_SECS)

    def update_plot(self):
        # Redibujar líneas
        for k in SENSOR_NAMES:
            self.lines[k].set_data(self.x, self.data[k])

        # Autoscale
        if len(self.x) > 2:
            self.ax.set_xlim(min(self.x), max(self.x))
            vals = []
            for k in SENSOR_NAMES:
                vals += [v for v in self.data[k] if v == v]
            if vals:
                ymin, ymax = min(vals), max(vals)
                pad = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
                self.ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas.draw_idle()
        self.root.after(1000, self.update_plot)

    def stop(self):
        self._stop = True

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()
