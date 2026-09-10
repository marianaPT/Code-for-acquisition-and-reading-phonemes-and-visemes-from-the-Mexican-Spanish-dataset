"""
Genera la nube de puntos 3D REAL de los landmarks de la boca.

Orden correcto:
  1. Detectar TODOS los landmarks sobre la imagen RGB ORIGINAL
     (sin recortar ni reescalar).
  2. Muestrear la profundidad en esas mismas coordenadas sobre
     depthAligned ORIGINAL (alineado pixel a pixel con el RGB,
     sin recortar).
  3. Proyectar a 3D real (X, Y, Z) usando los parametros intrinsecos
     de tu calibracion -- validos porque se aplican sobre la imagen
     en su resolucion nativa, antes de cualquier crop/resize.
  4. Quedarse con TODOS los landmarks que caen dentro del recuadro
     de la boca (el mismo bbox que usa crop_mouth_kinect.py para
     generar el recorte 256x256) -- no solo los indices de LIPS_IDX.
     Esto incluye menton, base de nariz, mejillas, lo que alcance a
     entrar en ese encuadre, igual que se ve en tus recortes RGB.

El recorte a 256x256 (crop_mouth_kinect.py) sigue sirviendo para
guardar las imagenes RGB/depth recortadas para inspeccion visual,
pero la nube de puntos se calcula aparte, sobre las coordenadas
originales, para que la geometria 3D sea correcta.
"""

import os
import glob
import csv

import numpy as np

from crop_mouth_kinect import (
    load_frame_from_mat,
    detect_all_landmarks,
    bbox_from_landmarks,
    LIPS_IDX,
)

MARGIN_RATIO = 0.35  # mismo margen que usa crop_mouth_kinect.py para el bbox

# --- PARAMETROS INTRINSECOS DE LA CAMARA (de kinect_calibration.mat) ---
# Corresponden a la camara de COLOR (colorFx/Fy/Cx/Cy), porque
# depthAligned ya viene registrado al espacio de la camara de color
# (alineacion colorCentric via pcfromkinect). Validos para la imagen
# ORIGINAL en 1920x1080.
FX = 1060.0644430230925   # distancia focal en x (pixeles)
FY = 1060.0443113213823   # distancia focal en y (pixeles)
CX = 959.9488523330053    # punto principal x (pixeles)
CY = 539.9687993207751    # punto principal y (pixeles)

DEPTH_SCALE_MM = 1000.0  # depthAligned viene en metros; se guarda en mm

# --- CONFIGURACION GENERAL ---
RUTAS_ENTRADA = [
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_A\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_A\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_A\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_BE\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_BE\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_BE\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_CHE\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_CHE\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_CHE\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_DA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_DA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_DA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_E\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_E\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_E\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_FA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_FA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_FA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_GA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_GA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_GA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_I\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_I\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_I\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_JA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_JA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_JA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_KA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_KA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_KA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_LA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_LA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_LA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_MA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_MA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_MA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_ÑA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_ÑA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_ÑA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_NO\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_NO\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_NO\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_O\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_O\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_O\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_PA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_PA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_PA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_RA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_RA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_RA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_SI\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_SI\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_SI\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_TE\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_TE\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_TE\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_U\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_U\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_U\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_YA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_YA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas_COMPLETO\Fonema_YA\P1\ENSAYO3\frames_aligned",
]
# --------------------


def pixel_a_3d(u, v, z, fx, fy, cx, cy):
    """Retroproyecta (u, v, z) -> (X, Y, Z) reales, modelo pinhole.

    Y se invierte (-1) porque en la imagen v crece hacia abajo
    (convencion de vision por computadora), mientras que en un visor
    3D estandar Y positivo se espera hacia arriba. Sin este ajuste,
    la nube sale "de cabeza" (menton arriba, nariz abajo).
    """
    Z = z
    X = (u - cx) * Z / fx
    Y = -(v - cy) * Z / fy
    return X, Y, Z


def landmarks_dentro_recuadro(rgb_full, depth_aligned_m, margin_ratio=MARGIN_RATIO):
    """PASO 1: Detecta los landmarks en la imagen RGB ORIGINAL y se
    queda con TODOS los que caen dentro del recuadro de la boca (mismo
    bbox que usa crop_mouth_kinect.py para el recorte 256x256) -- no
    solo los indices de LIPS_IDX. Para cada uno, muestrea su
    profundidad real en depthAligned ORIGINAL (sin recortar).

    Estos son los "landmarks" propiamente dichos -- posicion de pixel
    + profundidad -- antes de proyectarlos a 3D. Equivalen a los
    landmarks_rgb.txt / landmarks_depth.txt de antes, pero con todos
    los puntos del recuadro en vez de solo labios.

    Devuelve una lista de dicts: {"id", "x", "y", "depth_mm", "es_labio"}
    """
    h_rgb, w_rgb = rgb_full.shape[:2]
    all_points = detect_all_landmarks(rgb_full)
    if all_points is None:
        return None

    # El bbox se calcula a partir de los landmarks de labios (igual que
    # en crop_mouth_kinect.py), pero solo para definir el recuadro --
    # el filtro final usa TODOS los landmarks que caen dentro de el.
    lips_points = [p for p in all_points if p["id"] in LIPS_IDX]
    if not lips_points:
        return None

    x0, y0, box_w, box_h = bbox_from_landmarks(
        lips_points, w_rgb, h_rgb, margin_ratio=margin_ratio
    )
    x_min, y_min = x0, y0
    x_max, y_max = x0 + box_w, y0 + box_h

    h_d, w_d = depth_aligned_m.shape[:2]
    landmarks = []

    for p in all_points:
        u, v = p["x"], p["y"]

        # Filtro por posicion dentro del recuadro (en vez de LIPS_IDX)
        if not (x_min <= u <= x_max and y_min <= v <= y_max):
            continue

        ui = int(np.clip(round(u), 0, w_d - 1))
        vi = int(np.clip(round(v), 0, h_d - 1))

        z = depth_aligned_m[vi, ui]
        if np.isnan(z) or z <= 0:
            continue

        landmarks.append({
            "id": p["id"], "x": u, "y": v,
            "depth_mm": float(z) * DEPTH_SCALE_MM,
            "es_labio": p["id"] in LIPS_IDX,
        })

    return landmarks


def proyectar_a_nube(landmarks, fx, fy, cx, cy):
    """PASO 2: Toma los landmarks (x, y, depth_mm) ya extraidos y los
    proyecta a la nube de puntos 3D real (X, Y, Z), usando los
    parametros intrinsecos de la camara.

    Devuelve una lista de dicts: {"id", "X", "Y", "Z", "es_labio"}
    """
    puntos = []
    for lm in landmarks:
        X, Y, Z = pixel_a_3d(lm["x"], lm["y"], lm["depth_mm"], fx, fy, cx, cy)
        puntos.append({
            "id": lm["id"], "X": X, "Y": Y, "Z": Z,
            "es_labio": lm["es_labio"],
        })
    return puntos


def guardar_landmarks_txt(landmarks, path):
    """Guarda los landmarks (paso 1) como x,y,depth_mm -- mismo formato
    que tus archivos *_landmarks_depth_filtrada.txt de antes."""
    with open(path, "w") as f:
        for lm in landmarks:
            f.write(f"{lm['x']:.2f},{lm['y']:.2f},{lm['depth_mm']:.2f}\n")


def guardar_nube_txt(puntos, path):
    """Guarda la nube de puntos (paso 2) como X,Y,Z reales en mm."""
    with open(path, "w") as f:
        for p in puntos:
            f.write(f"{p['X']:.2f},{p['Y']:.2f},{p['Z']:.2f}\n")


def procesar_carpeta(input_folder):
    base_dir = os.path.dirname(input_folder)
    landmarks_folder = os.path.join(base_dir, "landmarks_ajustados")
    nube_folder = os.path.join(base_dir, "nube_puntos", "nube_landmarks_corregido")
    os.makedirs(landmarks_folder, exist_ok=True)
    os.makedirs(nube_folder, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))
    print(f"\n[PROCESANDO] {input_folder}")
    print(f"Encontrados {len(mat_files)} archivos .mat")

    log_rows = []
    n_ok, n_fail = 0, 0

    for i, mat_path in enumerate(mat_files, start=1):
        name = os.path.splitext(os.path.basename(mat_path))[0]

        try:
            rgb_full, depth_aligned_m = load_frame_from_mat(mat_path)
        except Exception as e:
            log_rows.append([name, "error_carga", str(e)])
            n_fail += 1
            continue

        # PASO 1: landmarks (pixel + profundidad) dentro del recuadro
        landmarks = landmarks_dentro_recuadro(rgb_full, depth_aligned_m)

        if not landmarks:
            log_rows.append([name, "sin_deteccion_o_sin_profundidad_valida", ""])
            n_fail += 1
            continue

        landmarks_out = os.path.join(landmarks_folder, f"{name}_landmarks.txt")
        guardar_landmarks_txt(landmarks, landmarks_out)

        # PASO 2: proyeccion de esos landmarks a la nube de puntos 3D
        puntos = proyectar_a_nube(landmarks, FX, FY, CX, CY)
        nube_out = os.path.join(nube_folder, f"{name}_nube_landmarks.txt")
        guardar_nube_txt(puntos, nube_out)

        log_rows.append([
            name, "ok",
            f"{len(landmarks)} landmarks en el recuadro "
            f"({sum(1 for lm in landmarks if lm['es_labio'])} son de labios)",
        ])
        n_ok += 1

        if i % 50 == 0:
            print(f"  {i} / {len(mat_files)} procesados")

    log_path = os.path.join(base_dir, "_log_nube_landmarks.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "estado", "detalle"])
        writer.writerows(log_rows)

    print(f"-> Finalizado. OK: {n_ok} | Fallidos: {n_fail}")
    print(f"-> Log guardado en: {log_path}")


def main():
    if FX == 0.0 or FY == 0.0:
        print("[ADVERTENCIA] FX/FY siguen en 0.0 - reemplaza los parametros "
              "de tu calibracion real antes de correr el batch.")
        return

    total_carpetas = len(RUTAS_ENTRADA)
    print(f"Iniciando procesamiento por lotes para {total_carpetas} carpetas.")

    for idx, ruta in enumerate(RUTAS_ENTRADA, start=1):
        print(f"\n================ Carpeta {idx} de {total_carpetas} ================")
        if os.path.exists(ruta):
            procesar_carpeta(ruta)
        else:
            print(f"[ERROR] La ruta no existe: {ruta}")

    print("\n==================================================")
    print("Procesamiento de todas las rutas completado.")


if __name__ == "__main__":
    main()