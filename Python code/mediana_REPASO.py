"""
Batch runner para filtrado de profundidad: recorre múltiples carpetas de 
RecortesBoca, detecta los archivos '_depth.png' y aplica el filtro de outliers.
Guarda los resultados un nivel arriba (al nivel del ENSAYO).
-----------------------------------------
Se usa un relleno iterativo: en cada pasada se corrigen únicamente
los píxeles atípicos que ya tienen al menos un vecino válido (ventana 3x3),
usando la mediana de esos vecinos, y se marcan como válidos para la
siguiente pasada. Esto hace que la corrección "crezca" desde el borde de
la región dañada hacia adentro, sin importar qué tan grande sea la región,
y SIN tocar nunca los píxeles que ya estaban dentro de rango.
"""

import os
import glob
import cv2
import numpy as np

# --- CONFIGURACIÓN DEL FILTRO ---
MIN_VALID, MAX_VALID = 300, 1000
MAX_ITER = 200  # límite de seguridad; en la práctica se resuelve en pocas decenas

# --- LISTA DE CARPETAS DE RECORTES A PROCESAR ---
CARPETAS_RECORTES = [
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1\ENSAYO3\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1\ENSAYO1\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1\ENSAYO2\recortes_boca",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1\ENSAYO3\recortes_boca",
]
# ------------------------------------------------


def _dilatar_mascara(mask):
    """Dilatación 3x3 pura con numpy (sin dependencias extra)."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            dilated |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return dilated


def corregir_outliers(depth_image, min_valid=MIN_VALID, max_valid=MAX_VALID,
                       max_iter=MAX_ITER):
    """
    Corrige únicamente los píxeles fuera de rango [min_valid, max_valid],
    usando la mediana de sus vecinos válidos más cercanos (ventana 3x3).
    Los píxeles ya válidos NUNCA se modifican.

    Funciona por "crecimiento": en cada pasada se resuelven los outliers que
    tocan al menos un vecino válido, y se van sumando a la región válida.
    Así se corrigen regiones de cualquier tamaño (un par de píxeles sueltos
    o un bloque grande), siempre que exista algo de dato válido en la imagen.
    """
    depth = depth_image.astype(np.float64)
    valid_now = (depth >= min_valid) & (depth <= max_valid)
    faltantes = ~valid_now
    resultado = depth.copy()

    if not faltantes.any():
        return depth_image.copy(), 0, 0  # nada que corregir

    it = 0
    while faltantes.any() and it < max_iter:
        it += 1
        dilatada = _dilatar_mascara(valid_now)
        frontera = faltantes & dilatada
        if not frontera.any():
            # No hay forma de seguir creciendo (no queda dato válido cerca)
            break

        ys, xs = np.where(frontera)
        nuevos_valores = {}
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(resultado.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(resultado.shape[1], x + 2)
            ventana_vals = resultado[y0:y1, x0:x1]
            ventana_validos = valid_now[y0:y1, x0:x1]
            vecinos_validos = ventana_vals[ventana_validos]
            nuevos_valores[(y, x)] = np.median(vecinos_validos)

        for (y, x), v in nuevos_valores.items():
            resultado[y, x] = v
            valid_now[y, x] = True
            faltantes[y, x] = False

    sin_resolver = int(faltantes.sum())
    return resultado.astype(depth_image.dtype), it, sin_resolver


def filtrar_imagen_profundidad(depth_path, output_folder):
    """Procesa una sola imagen de profundidad y guarda el resultado."""
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_image is None:
        return False

    filtered_depth_16, iteraciones, sin_resolver = corregir_outliers(depth_image)

    if sin_resolver > 0:
        # No había ningún dato válido cerca para resolver estos píxeles
        # (imagen probablemente inválida en su totalidad, o corrupta).
        print(f"  [AVISO] {os.path.basename(depth_path)}: "
              f"{sin_resolver} píxeles no se pudieron corregir "
              f"(sin datos válidos cercanos).")

    # Generar el nombre de salida (ej: color_0051_depth_filtrada.png)
    base_name = os.path.basename(depth_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_filtrada{ext}"
    output_path = os.path.join(output_folder, output_filename)

    # Guardar el resultado en la carpeta destino
    success = cv2.imwrite(output_path, filtered_depth_16)
    return success


def procesar_lote_filtrado(input_folder):
    """Busca imágenes depth en RecortesBoca y las guarda al nivel de ENSAYO."""
    # os.path.dirname obtiene la ruta del ENSAYO (sube un nivel desde RecortesBoca)
    ensayo_folder = os.path.dirname(input_folder)
    output_folder = os.path.join(ensayo_folder, "resultados_filtrados")
    os.makedirs(output_folder, exist_ok=True)

    # Busca únicamente los archivos que terminen en _depth.png
    depth_files = sorted(glob.glob(os.path.join(input_folder, "*_depth.png")))

    print(f"\n[FILTRANDO] {input_folder}")
    print(f"Encontradas {len(depth_files)} imágenes de profundidad.")

    n_ok = 0
    for i, depth_path in enumerate(depth_files, start=1):
        success = filtrar_imagen_profundidad(depth_path, output_folder)
        if success:
            n_ok += 1

        if i % 50 == 0:
            print(f"  {i} / {len(depth_files)} imágenes filtradas...")

    print(f"-> Completado. Guardadas {n_ok} de {len(depth_files)} imágenes.")
    print(f"-> Destino: {output_folder}")


def main():
    total_carpetas = len(CARPETAS_RECORTES)
    print(f"Iniciando lote de filtrado para {total_carpetas} carpetas.")

    for idx, carpeta in enumerate(CARPETAS_RECORTES, start=1):
        print(
            f"\n================ Carpeta {idx} de {total_carpetas} ================"
        )
        if os.path.exists(carpeta):
            procesar_lote_filtrado(carpeta)
        else:
            print(f"[ERROR] La ruta no existe: {carpeta}")

    print("\n==================================================")
    print("¡El proceso de filtrado masivo ha terminado!")


if __name__ == "__main__":
    main()