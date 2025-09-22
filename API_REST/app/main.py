from __future__ import annotations
import os, time, math, threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional, List, Literal, Dict, Union

import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, conlist

# Arduino IoT Cloud
import iot_api_client as iot
from iot_api_client.configuration import Configuration
from iot_api_client.rest import ApiException
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Config Arduino IoT + ML
CLIENT_ID     = os.getenv("ARDUINO_CLIENT_ID", "9edQkqqI6Ti9RnF1HuUcMHd9plc21mBW")
CLIENT_SECRET = os.getenv("ARDUINO_CLIENT_SECRET", "CNm0vxo90QHg3VitUgv3o9yNy0EVsqguvteaiVphvR1pfBgQ88xAC2p0cMYkoDYu")
THING_ID      = os.getenv("ARDUINO_THING_ID", "38fbfa60-5c0b-47e8-aa56-2f5083ffb631")

API_HOST   = "https://api2.arduino.cc"
TOKEN_URL  = f"{API_HOST}/iot/v1/clients/token"
AUDIENCE   = f"{API_HOST}/iot"

# Sensores esperados (nombres de propiedades en el Thing)
SENSOR_KEYS = ["temp", "hum", "air_q", "light"]

# Buffer de datos en memoria
WINDOW_SECONDS = int(os.getenv("IOT_WINDOW_SECONDS", "3600"))  # 1h
POLL_SECONDS   = int(os.getenv("IOT_POLL_SECONDS", "1"))      

# Modelo ML
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_TASK: Literal["regression","classification"] = os.getenv("MODEL_TASK", "regression")
PREDICT_TARGET = os.getenv("PREDICT_TARGET", "air_q_next")

# Umbrales para alertas
THRESHOLDS = {
    "temp":  {"low": 18.0, "high": 30.0},
    "hum":   {"low": 30.0, "high": 70.0},
    "air_q": {"high": 1000.0, "crit_high": 1500.0},
    "light": {"low": 100.0}
}

# Modelos Pydantic
class Reading(BaseModel):
    timestamp: float = Field(..., description="Unix seconds")
    temp: Optional[float] = None
    hum: Optional[float] = None
    air_q: Optional[float] = None
    light: Optional[float] = None

class PredictIn(BaseModel):
    temp: float
    hum: float
    air_q: float
    light: float

class PredictOut(BaseModel):
    task: str
    target: str
    y_pred: float
    fallback: bool = False
    used_source: Literal["manual","iot_buffer"]

class TrendOut(BaseModel):
    key: str
    slope_per_min: float
    intercept: float
    r2: float
    direction: Literal["rising","falling","flat"]
    n_points: int

class AlertsOut(BaseModel):
    alerts: List[Dict]
    zscores: Dict[str, float]
    window_size: int

class ControlIn(BaseModel):
    property_name: Literal["heat_lamp","water"]
    # acepta true/false o 0/1
    value: Union[bool, int] = Field(..., description="0/1 o true/false")

# cliente Arduino IoT Cloud
class ArduinoIoTClient:
    def __init__(self, client_id: str, client_secret: str, thing_id: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.thing_id = thing_id
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        self._api_client: Optional[iot.ApiClient] = None
        self._lock = threading.Lock()

    def _fetch_token(self) -> str:
        oauth_client = BackendApplicationClient(client_id=self.client_id)
        oauth = OAuth2Session(client=oauth_client)
        token = oauth.fetch_token(
            token_url=TOKEN_URL,
            client_id=self.client_id,
            client_secret=self.client_secret,
            include_client_id=True,
            audience=AUDIENCE,
        )
        # Arduino devuelve exp ~3600s; refrescamos 1 min antes
        self._token_expiry = time.time() + float(token.get("expires_in", 3600)) - 60
        return token.get("access_token")

    def _ensure_client(self):
        with self._lock:
            if self._access_token is None or time.time() >= self._token_expiry:
                self._access_token = self._fetch_token()
                cfg = Configuration(host=API_HOST)
                cfg.access_token = self._access_token
                self._api_client = iot.ApiClient(cfg)

    def properties_api(self) -> iot.PropertiesV2Api:
        self._ensure_client()
        return iot.PropertiesV2Api(self._api_client)

    def get_latest(self) -> dict:
        """
        Devuelve dict con:
          - numeric: {temp, hum, air_q, light}
          - textual: {otras propiedades string}
          - timestamp: Unix seconds (server time)
        """
        api = self.properties_api()
        resp = api.properties_v2_list(self.thing_id)
        numeric, textual = {}, {}
        for prop in resp:
            name = prop.name
            val  = prop.last_value
            # intenta convertir a float
            try:
                fval = float(val)
                numeric[name] = fval
            except Exception:
                textual[name] = val
        now = time.time()
        return {"timestamp": now, "numeric": numeric, "textual": textual}

    def update_property(self, property_name: str, value):
        api = self.properties_api()
        # Arduino SDK: properties_v2_publish(thing_id, property_id, property_value)
        # Necesitamos property_id -> lo resolvemos listando y buscando por name.
        props = api.properties_v2_list(self.thing_id)
        prop_id = None
        for p in props:
            if p.name == property_name:
                prop_id = p.id
                break
        if prop_id is None:
            raise HTTPException(404, f"Propiedad '{property_name}' no encontrada en el Thing")

        body = {"value": value}
        api.properties_v2_publish(self.thing_id, prop_id, body)

# Management del buffer de datos
class RingBuffer:
    def __init__(self, max_seconds: int):
        self.max_seconds = max_seconds
        self._buf: deque[Reading] = deque()
        self._lock = threading.Lock()

    def add(self, r: Reading):
        with self._lock:
            self._buf.append(r)
            self._trim()

    def _trim(self):
        # elimina más antiguos fuera de ventana [now - max_seconds, now]
        now = time.time()
        while self._buf and (now - self._buf[0].timestamp) > self.max_seconds:
            self._buf.popleft()

    def snapshot(self) -> List[Reading]:
        with self._lock:
            return list(self._buf)

    def last(self) -> Optional[Reading]:
        with self._lock:
            return self._buf[-1] if self._buf else None

# Utilidades para el modelo ML
def _load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

MODEL = _load_model()

def _feature_vector_from_reading(r: Reading) -> np.ndarray:
    # Orden esperado: temp, hum, air_q, light
    return np.array([[r.temp, r.hum, r.air_q, r.light]], dtype=float)

def _feature_vector_from_predictin(x: PredictIn) -> np.ndarray:
    return np.array([[x.temp, x.hum, x.air_q, x.light]], dtype=float)

def _fallback_formula(temp, hum, air_q, light) -> float:
    # Naive combinación
    return 0.5*air_q + 0.3*(temp*20) + 0.15*(100 - hum) + 0.05*(max(0.0, 1200 - light)/10.0)

def _trend_linreg(t_min: np.ndarray, y: np.ndarray):
    t = t_min.reshape(-1,1)
    XTX = t.T @ t
    if XTX[0,0] == 0:
        return 0.0, float(y.mean()), 0.0
    a = float(np.linalg.inv(XTX) @ t.T @ y)  # slope
    b = float(y.mean() - a * t.mean())
    y_hat = a * t.flatten() + b
    ss_res = float(np.sum((y - y_hat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return a, b, r2

def _alerts_from_window(window: List[Reading], current: Reading) -> AlertsOut:
    alerts: List[Dict] = []
    zscores: Dict[str,float] = {}

    # 1) Umbrales
    for k, conf in THRESHOLDS.items():
        v = getattr(current, k, None)
        if v is None:
            continue
        if "low" in conf and v < conf["low"]:
            level = "crit" if ("crit_low" in conf and v < conf["crit_low"]) else "warn"
            alerts.append({"metric": k, "level": level, "reason": f"{k} < {conf['low']}"})
        if "high" in conf and v > conf["high"]:
            level = "crit" if ("crit_high" in conf and v > conf["crit_high"]) else "warn"
            alerts.append({"metric": k, "level": level, "reason": f"{k} > {conf['high']}"})

    # 2) Z-score
    for k in ["temp","hum","air_q","light"]:
        arr = [getattr(r, k) for r in window if getattr(r, k) is not None]
        arr = [a for a in arr if not (isinstance(a, float) and math.isnan(a))]
        if len(arr) >= 5:
            mu = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            val = getattr(current, k, None)
            if val is None or sd == 0:
                z = 0.0
            else:
                z = (val - mu) / sd
            zscores[k] = float(z)
            if abs(z) >= 3.0:
                alerts.append({"metric": k, "level":"crit", "reason": f"Anomalía (|z|={abs(z):.2f}≥3)"})
            elif abs(z) >= 2.0:
                alerts.append({"metric": k, "level":"warn", "reason": f"Posible anomalía (|z|={abs(z):.2f}≥2)"})
        else:
            zscores[k] = 0.0

    return AlertsOut(alerts=alerts, zscores=zscores, window_size=len(window))

# API REST con FastAPI
app = FastAPI(title="Env ML API + Arduino IoT Cloud", version="1.1.0")

IOT = ArduinoIoTClient(CLIENT_ID, CLIENT_SECRET, THING_ID)
BUFFER = RingBuffer(WINDOW_SECONDS)

def _reading_from_numeric_dict(ts: float, numeric: dict) -> Reading:
    # Garantiza None si clave no está
    vals = {k: (float(numeric[k]) if k in numeric else None) for k in SENSOR_KEYS}
    return Reading(timestamp=ts, **vals)

def _poll_once_and_store():
    try:
        latest = IOT.get_latest()
        ts = float(latest["timestamp"])
        r = _reading_from_numeric_dict(ts, latest["numeric"])
        BUFFER.add(r)
        return latest, r
    except ApiException as e:
        raise HTTPException(502, f"Arduino IoT API error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Error leyendo IoT: {e}")

# Background poller
_stop_flag = False
def _background_poller():
    while not _stop_flag:
        try:
            _poll_once_and_store()
        except Exception:
            pass
        time.sleep(POLL_SECONDS)

@app.on_event("startup")
def _on_start():
    # hace un primer pull para inicializar
    try:
        _poll_once_and_store()
    except Exception:
        pass
    # lanza thread de polling
    t = threading.Thread(target=_background_poller, daemon=True)
    t.start()

@app.on_event("shutdown")
def _on_stop():
    global _stop_flag
    _stop_flag = True

# ENDPOINTS

@app.get("/health")
def health():
    last = BUFFER.last()
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "thing_id": THING_ID,
        "buffer_points": len(BUFFER.snapshot()),
        "last_ts": (datetime.fromtimestamp(last.timestamp).isoformat() if last else None)
    }

@app.get("/iot/latest")
def iot_latest():
    """
    Devuelve último snapshot directamente de Arduino IoT + textual props.
    No depende del background poller.
    """
    latest = IOT.get_latest()
    return latest

@app.post("/iot/pull")
def iot_pull():
    """
    Fuerza un pull inmediato y guarda en buffer.
    """
    latest, r = _poll_once_and_store()
    return {"stored": True, "reading": r, "textual": latest["textual"]}

@app.get("/iot/buffer")
def iot_buffer(limit: int = Query(120, ge=1, le=5000)):
    """
    Devuelve últimos 'limit' elementos del buffer.
    """
    snap = BUFFER.snapshot()[-limit:]
    return {"count": len(snap), "readings": snap}

@app.post("/predict", response_model=PredictOut)
def predict(x: Optional[PredictIn] = None, use_iot_latest: bool = Query(False)):
    """
    Predicción ambiental:
    - Si use_iot_latest=True, usa el último dato del buffer IoT (hace pull si buffer vacío).
    - Si envías cuerpo JSON, usa esos valores manualmente.
    """
    used_source = "manual"
    if use_iot_latest:
        last = BUFFER.last()
        if last is None:
            # intenta un pull
            _poll_once_and_store()
            last = BUFFER.last()
            if last is None:
                raise HTTPException(400, "No hay datos en el buffer IoT")
        X = _feature_vector_from_reading(last)
        used_source = "iot_buffer"
    else:
        if x is None:
            raise HTTPException(422, "Debes enviar PredictIn o usar use_iot_latest=true")
        X = _feature_vector_from_predictin(x)

    if MODEL is not None:
        y = float(MODEL.predict(X)[0])
        return PredictOut(task=MODEL_TASK, target=PREDICT_TARGET, y_pred=y, fallback=False, used_source=used_source)

    # fallback
    if use_iot_latest:
        temp, hum, air_q, light = X.flatten().tolist()
    else:
        temp, hum, air_q, light = x.temp, x.hum, x.air_q, x.light
    y = _fallback_formula(temp, hum, air_q, light)
    return PredictOut(task="regression", target=PREDICT_TARGET, y_pred=float(y), fallback=True, used_source=used_source)

@app.get("/trend", response_model=TrendOut)
def trend(key: Literal["temp","hum","air_q","light"], minutes: int = Query(30, ge=3, le=1440)):
    """
    Tendencia a partir del buffer IoT en los últimos 'minutes'.
    """
    snap = BUFFER.snapshot()
    if not snap:
        # intenta poblar
        _poll_once_and_store()
        snap = BUFFER.snapshot()
    # filtra por ventana temporal
    now = time.time()
    window = [r for r in snap if (now - r.timestamp) <= minutes*60]
    y = [getattr(r, key) for r in window if getattr(r, key) is not None]
    t = [ (r.timestamp - window[0].timestamp)/60.0 for r in window ] if window else []

    if len(y) < 3:
        raise HTTPException(400, "Se requieren >=3 puntos en ventana")

    y = np.array(y, dtype=float)
    t = np.array(t, dtype=float)
    slope, intercept, r2 = _trend_linreg(t, y)

    eps = max(1e-9, 0.01 * np.std(y) if np.std(y)>0 else 0.01)
    direction = "flat"
    if slope > eps: direction = "rising"
    elif slope < -eps: direction = "falling"

    return TrendOut(key=key, slope_per_min=float(slope), intercept=float(intercept), r2=float(r2), direction=direction, n_points=len(y))

@app.get("/alerts/evaluate", response_model=AlertsOut)
def alerts_evaluate(minutes: int = Query(30, ge=5, le=1440)):
    """
    Evalúa alertas usando:
      - Ventana reciente del buffer IoT (últimos 'minutes')
      - Punto actual: último reading del buffer (hace pull si necesario)
    """
    snap = BUFFER.snapshot()
    if not snap:
        _poll_once_and_store()
        snap = BUFFER.snapshot()
    now = time.time()
    window = [r for r in snap if (now - r.timestamp) <= minutes*60]
    if len(window) < 5:
        raise HTTPException(400, "Ventana insuficiente para evaluar alertas (>=5 puntos)")

    current = window[-1]
    return _alerts_from_window(window, current)

@app.post("/iot/control")
def iot_control(body: ControlIn):
    """
    Publica un valor en una propiedad del Thing (p.ej., heat_lamp/water).
    """
    try:
        value = int(body.value) if isinstance(body.value, bool) else body.value
        IOT.update_property(body.property_name, value)
        return {"ok": True, "property": body.property_name, "value": value}
    except HTTPException:
        raise
    except ApiException as e:
        raise HTTPException(502, f"Arduino IoT API error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Error de control: {e}")
