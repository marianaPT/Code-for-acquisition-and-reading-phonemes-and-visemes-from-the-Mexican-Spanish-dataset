"""
Batch runner multi-ruta: recorre varias carpetas de .mat y genera
los recortes de boca en 256x256 para cada frame, junto con los
landmarks (RGB y depth) y una imagen de verificación (overlay).
Los archivos .mat son los generados por el archivo de matlab 
aling_frames_many_ejecutions.m.

Estructura de salida (a partir de la carpeta ENSAYO):

    ENSAYO1/
        frames_aligned/           (entrada, ya existente)
        RecortesBoca/
            {name}_rgb.png
            {name}_depth.png
        landmarks/
            RGB/                  {name}_landmarks_rgb.txt
            Profundidad/          {name}_landmarks_depth.txt
        overlay/                  {name}_overlay.png
"""

import os
import glob
import csv

from crop_mouth_kinect import (
    load_frame_from_mat,
    process_frame_with_landmarks,
    save_rgb_png,
    save_depth_png16,
    save_landmarks_txt,
    visualize_landmarks_overlay,
)

# --- CONFIGURACIÓN GENERAL ---
SIZE = 256
MARGIN_RATIO = 0.35
GUARDAR_OVERLAY = True  # ponlo en False si no quieres la imagen de verificación (más rápido)

# --- LISTA DE RUTAS A PROCESAR ---
RUTAS_ENTRADA = [
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1\ENSAYO3\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1\ENSAYO1\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1\ENSAYO2\frames_aligned",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1\ENSAYO3\frames_aligned",
]
# --------------------

def procesar_carpeta(input_folder):
    # base_dir es la carpeta ENSAYO (un nivel arriba de 'frames_aligned')
    base_dir = os.path.dirname(input_folder)

    # RecortesBoca solo contiene las imágenes, separadas en RGB/ y Profundidad/
    recortes_folder = os.path.join(base_dir, "recortes_boca")

    # landmarks y overlay quedan al nivel de ENSAYO
    landmarks_rgb_folder = os.path.join(base_dir, "landmarks", "rgb")
    landmarks_depth_folder = os.path.join(base_dir, "landmarks", "profundidad")
    overlay_folder = os.path.join(base_dir, "overlay")

    os.makedirs(recortes_folder, exist_ok=True)
    os.makedirs(landmarks_rgb_folder, exist_ok=True)
    os.makedirs(landmarks_depth_folder, exist_ok=True)
    if GUARDAR_OVERLAY:
        os.makedirs(overlay_folder, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))
    print(f"\n[PROCESANDO] {input_folder}")
    print(f"Encontrados {len(mat_files)} archivos .mat")

    log_rows = []
    n_ok, n_fail = 0, 0

    for i, mat_path in enumerate(mat_files, start=1):
        name = os.path.splitext(os.path.basename(mat_path))[0]

        try:
            rgb, depth_aligned = load_frame_from_mat(mat_path)
        except Exception as e:
            log_rows.append([name, "error_carga", str(e)])
            n_fail += 1
            continue

        result = process_frame_with_landmarks(
            rgb, depth_aligned, size=SIZE, margin_ratio=MARGIN_RATIO
        )

        if result is None:
            log_rows.append([name, "sin_deteccion", ""])
            n_fail += 1
            continue

        rgb_crop, depth_crop, rgb_landmarks, depth_landmarks = result

        rgb_out = os.path.join(recortes_folder, f"{name}_rgb.png")
        depth_out = os.path.join(recortes_folder, f"{name}_depth.png")
        rgb_landmarks_out = os.path.join(landmarks_rgb_folder, f"{name}_landmarks_rgb.txt")
        depth_landmarks_out = os.path.join(landmarks_depth_folder, f"{name}_landmarks_depth.txt")

        save_rgb_png(rgb_crop, rgb_out)
        save_depth_png16(depth_crop, depth_out)
        save_landmarks_txt(rgb_landmarks, rgb_landmarks_out)
        save_landmarks_txt(depth_landmarks, depth_landmarks_out)

        if GUARDAR_OVERLAY:
            overlay_out = os.path.join(overlay_folder, f"{name}_overlay.png")
            visualize_landmarks_overlay(
                rgb_crop, depth_crop, rgb_landmarks, depth_landmarks, overlay_out
            )

        log_rows.append([name, "ok", ""])
        n_ok += 1

        if i % 50 == 0:
            print(f"  {i} / {len(mat_files)} procesados")

    # Guardar log (al nivel de ENSAYO, junto a landmarks/overlay)
    log_path = os.path.join(base_dir, "_log_procesamiento.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "estado", "detalle"])
        writer.writerows(log_rows)

    print(f"-> Finalizado. OK: {n_ok} | Fallidos: {n_fail}")
    print(f"-> Log guardado en: {log_path}")


def main():
    total_carpetas = len(RUTAS_ENTRADA)
    print(f"Iniciando procesamiento por lotes para {total_carpetas} carpetas.")

    for idx, ruta in enumerate(RUTAS_ENTRADA, start=1):
        print(f"\n================ Carpeta {idx} de {total_carpetas} ================")
        if os.path.exists(ruta):
            procesar_carpeta(ruta)
        else:
            print(f"[ERROR] La ruta no existe: {ruta}")

    print("\n==================================================")
    print("¡Procesamiento de todas las rutas completado con éxito!")

if __name__ == "__main__":
    main()