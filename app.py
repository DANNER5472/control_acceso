"""
Dashboard de control de acceso: YOLOv8 + InsightFace + Flask

Instalar:
    pip install flask ultralytics insightface onnxruntime opencv-python numpy

Correr:
    python app.py
    Abrir http://localhost:5000
"""
import os
import pickle
import sqlite3
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# ---------- configuracion ----------
DB_PATH = "registrados.pkl"    # nombre -> lista de embeddings
SQL_PATH = "eventos.db"
SNAP_DIR = "snapshots"
UMBRAL = 0.45                  # similitud coseno minima para "conocido"
FRAMES_ALARMA = 8              # frames seguidos con desconocido antes de alarmar
COOLDOWN_ALARMA = 5.0          # segundos entre alarmas
COOLDOWN_ACCESO = 30.0         # segundos para volver a registrar el acceso de la misma persona
CAMARA = 0

os.makedirs(SNAP_DIR, exist_ok=True)

# ---------- modelos ----------
yolo = YOLO("yolov8n.pt")
caras = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
caras.prepare(ctx_id=-1, det_size=(320, 320))
model_lock = threading.Lock()   # InsightFace lo usan la camara y el registro

# ---------- estado compartido ----------
estado_lock = threading.Lock()
estado = {"jpeg": None, "frame": None, "personas": 0, "conocidos": 0,
          "desconocidos": 0, "fps": 0.0, "alarma_hasta": 0.0}

db_lock = threading.Lock()
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        personas_db = pickle.load(f)
else:
    personas_db = {}


def guardar_personas():
    with open(DB_PATH, "wb") as f:
        pickle.dump(personas_db, f)


# ---------- eventos (SQLite) ----------
def iniciar_sql():
    con = sqlite3.connect(SQL_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS eventos(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT, tipo TEXT, nombre TEXT, similitud REAL, foto TEXT)""")
    con.commit()
    con.close()


def registrar_evento(tipo, nombre, sim, foto=None):
    con = sqlite3.connect(SQL_PATH)
    con.execute("INSERT INTO eventos(fecha,tipo,nombre,similitud,foto) VALUES(?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo, nombre, float(sim), foto))
    con.commit()
    con.close()


SONIDO = {"activo": True}


def sonar_alarma():
    """Sirena tipo policia: el tono sube y baja dos veces."""
    if not SONIDO["activo"]:
        return

    def _sirena():
        try:
            import winsound
            for _ in range(2):
                for f in range(600, 1500, 60):
                    winsound.Beep(f, 25)
                for f in range(1500, 600, -60):
                    winsound.Beep(f, 25)
        except ImportError:
            print("\a", end="", flush=True)
    threading.Thread(target=_sirena, daemon=True).start()


# ---------- reconocimiento ----------
def identificar(embedding):
    mejor_nombre, mejor_sim = None, -1.0
    with db_lock:
        for nombre, muestras in personas_db.items():
            sim = max(float(np.dot(embedding, m)) for m in muestras)
            if sim > mejor_sim:
                mejor_nombre, mejor_sim = nombre, sim
    if mejor_sim >= UMBRAL:
        return mejor_nombre, mejor_sim
    return None, mejor_sim


def bucle_camara():
    cap = cv2.VideoCapture(CAMARA)
    seguidos = 0
    ultima_alarma = 0.0
    ultimo_acceso = {}
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        with estado_lock:
            estado["frame"] = frame.copy()

        vista = frame.copy()
        h, w = frame.shape[:2]
        personas = conocidos = desconocidos = 0
        sim_desc = 0.0

        for box in yolo(frame, classes=[0], conf=0.5, verbose=False)[0].boxes:
            personas += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            with model_lock:
                detectadas = caras.get(frame[y1:y2, x1:x2])

            if not detectadas:
                color, texto = (0, 255, 255), "sin cara visible"
            else:
                nombre, sim = identificar(detectadas[0].normed_embedding)
                if nombre:
                    conocidos += 1
                    color, texto = (0, 255, 0), f"{nombre} ({sim:.2f})"
                    if time.time() - ultimo_acceso.get(nombre, 0) > COOLDOWN_ACCESO:
                        ultimo_acceso[nombre] = time.time()
                        registrar_evento("ACCESO", nombre, sim)
                else:
                    desconocidos += 1
                    sim_desc = sim
                    color, texto = (0, 0, 255), f"DESCONOCIDO ({sim:.2f})"

            cv2.rectangle(vista, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vista, texto, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        seguidos = seguidos + 1 if desconocidos else 0
        if seguidos >= FRAMES_ALARMA and time.time() - ultima_alarma > COOLDOWN_ALARMA:
            ultima_alarma = time.time()
            archivo = f"alarma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(SNAP_DIR, archivo), vista)
            registrar_evento("ALARMA", "Desconocido", sim_desc, archivo)
            sonar_alarma()
            with estado_lock:
                estado["alarma_hasta"] = time.time() + COOLDOWN_ALARMA

        ahora = time.time()
        fps = 1.0 / max(ahora - t_prev, 1e-3)
        t_prev = ahora
        ok, buf = cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with estado_lock:
            estado.update(personas=personas, conocidos=conocidos, desconocidos=desconocidos,
                          fps=round(fps, 1))
            if ok:
                estado["jpeg"] = buf.tobytes()


# ---------- web ----------
app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video")
def video():
    def generar():
        while True:
            with estado_lock:
                jpg = estado["jpeg"]
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)
    return Response(generar(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/estado")
def api_estado():
    with estado_lock:
        datos = {k: estado[k] for k in ("personas", "conocidos", "desconocidos", "fps")}
        datos["alarma"] = time.time() < estado["alarma_hasta"]
    with db_lock:
        datos["registrados"] = len(personas_db)
    datos["sonido"] = SONIDO["activo"]
    return jsonify(datos)


@app.post("/api/sonido")
def api_sonido():
    SONIDO["activo"] = bool((request.get_json(silent=True) or {}).get("activo", True))
    return jsonify(ok=True, activo=SONIDO["activo"])


@app.get("/api/eventos")
def api_eventos():
    limite = min(int(request.args.get("limit", 30)), 200)
    con = sqlite3.connect(SQL_PATH)
    filas = con.execute(
        "SELECT fecha,tipo,nombre,similitud,foto FROM eventos ORDER BY id DESC LIMIT ?",
        (limite,)).fetchall()
    con.close()
    return jsonify([dict(fecha=f, tipo=t, nombre=n, similitud=round(s, 2), foto=p)
                    for f, t, n, s, p in filas])


@app.get("/api/resumen")
def api_resumen():
    hoy = datetime.now().strftime("%Y-%m-%d")
    con = sqlite3.connect(SQL_PATH)
    filas = con.execute(
        "SELECT substr(fecha,12,2), tipo, COUNT(*) FROM eventos WHERE fecha LIKE ? GROUP BY 1,2",
        (hoy + "%",)).fetchall()
    con.close()
    accesos, alarmas = [0] * 24, [0] * 24
    for hora, tipo, n in filas:
        (accesos if tipo == "ACCESO" else alarmas)[int(hora)] = n
    return jsonify(accesos=accesos, alarmas=alarmas,
                   total_accesos=sum(accesos), total_alarmas=sum(alarmas))


@app.get("/api/personas")
def api_personas():
    with db_lock:
        return jsonify([dict(nombre=n, muestras=len(m)) for n, m in personas_db.items()])


@app.post("/api/registrar")
def api_registrar():
    nombre = ((request.get_json(silent=True) or {}).get("nombre") or "").strip()[:40]
    if not nombre:
        return jsonify(ok=False, error="Escribe un nombre"), 400
    with estado_lock:
        frame = None if estado["frame"] is None else estado["frame"].copy()
    if frame is None:
        return jsonify(ok=False, error="La camara aun no entrega imagen"), 503
    with model_lock:
        detectadas = caras.get(frame)
    if len(detectadas) != 1:
        return jsonify(ok=False,
                       error=f"Se detectaron {len(detectadas)} caras; debe haber exactamente 1"), 400
    with db_lock:
        personas_db.setdefault(nombre, []).append(detectadas[0].normed_embedding)
        guardar_personas()
        total = len(personas_db[nombre])
    return jsonify(ok=True, muestras=total)


@app.delete("/api/personas/<nombre>")
def api_borrar(nombre):
    with db_lock:
        personas_db.pop(nombre, None)
        guardar_personas()
    return jsonify(ok=True)


@app.get("/snapshots/<path:archivo>")
def snapshots(archivo):
    return send_from_directory(SNAP_DIR, archivo)


if __name__ == "__main__":
    iniciar_sql()
    threading.Thread(target=bucle_camara, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)