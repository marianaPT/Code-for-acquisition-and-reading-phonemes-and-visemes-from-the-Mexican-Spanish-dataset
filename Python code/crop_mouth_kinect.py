"""
Recorte alineado de RGB y Depth (Kinect v2) centrado en la boca, a 256x256.

Requiere que, durante la captura en MATLAB, hayas usado pcfromkinect con
alineación 'colorCentric' (el modo por defecto) para obtener el depth
ya remuestreado a la resolución de la imagen de color:

    ptCloud = pcfromkinect(depthDevice, depthImage, colorImage);
    depthAligned = ptCloud.Location(:,:,3);   % Z en metros, mismo tamaño que colorImage
    save('frame001.mat', 'colorImage', 'depthAligned');

Flujo:
    1. Detectar landmarks de labios en la imagen RGB con MediaPipe Face Mesh.
    2. Calcular el bounding box de esos landmarks (con margen) y hacerlo cuadrado.
    3. Recortar el mismo bbox en RGB y en depthAligned.
    4. Reescalar ambos a 256x256.
    5. Guardar RGB como PNG normal (BGR) y depth como PNG de 16 bits (mm).

Se ejecuta batch_crop_mouths.py.
"""

import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Índices de landmarks de labios (contorno externo + interno), derivados
# del conjunto de conexiones FACEMESH_LIPS.
_LIPS_EDGES = mp.solutions.face_mesh.FACEMESH_LIPS
LIPS_IDX = sorted({i for edge in _LIPS_EDGES for i in edge})

# Escala para guardar el depth como PNG de 16 bits (metros -> milímetros)
DEPTH_SCALE_MM = 1000.0


def load_frame_from_mat(mat_path):
    """Carga colorAligned y depthAligned desde un .mat exportado de MATLAB
    (align_frames.m). Usamos colorAligned (ptCloud.Color) y NO el
    colorImage crudo, porque pcfromkinect corrige el espejo de Kinect
    solo dentro de ptCloud -- el colorImage original queda desalineado
    respecto a depthAligned si se usa directamente.

    colorAligned queda en orden de canales RGB (igual que las imágenes
    en MATLAB). Se mantiene así durante todo el pipeline; solo se
    convierte a BGR justo antes de escribir con OpenCV (ver save_rgb_png).
    """
    from scipy.io import loadmat
    data = loadmat(mat_path)
    rgb = data["colorAligned"]
    depth_aligned = data["depthAligned"]
    return rgb, depth_aligned


def detect_all_landmarks(rgb_image):
    """Corre FaceMesh UNA vez sobre el frame completo y devuelve TODOS
    los landmarks del mesh (468 puntos base + 10 de iris, porque se usa
    refine_landmarks=True) en coordenadas de píxel del frame original,
    o None si no se detectó cara.

    A diferencia de detect_lips_landmarks, aquí no se filtra por índice
    -- el filtrado a "lo que cae dentro del recorte" se hace después,
    con landmarks_within_crop, una vez que ya sabes el bbox.
    """
    h, w = rgb_image.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        results = fm.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    points = []
    for idx, lm in enumerate(landmarks):
        points.append({
            "id": idx,
            "x": lm.x * w,
            "y": lm.y * h,
            "z_mediapipe": lm.z,
            "visibility": getattr(lm, "visibility", None),
        })
    return points


def landmarks_within_crop(points_crop_coords, size=256):
    """Filtra una lista de landmarks YA transformados al espacio del
    recorte (ver landmarks_to_crop_coords), quedándose solo con los que
    caen dentro de los límites 0..size-1 en ambos ejes -- es decir, los
    que efectivamente son visibles en la imagen recortada (boca,
    barbilla, parte de nariz, mejillas, lo que alcance a entrar)."""
    return [
        p for p in points_crop_coords
        if 0 <= p["x"] < size and 0 <= p["y"] < size
    ]


def detect_lips_landmarks(rgb_image):
    """Corre FaceMesh UNA vez sobre el frame completo y devuelve los
    landmarks de labios en coordenadas de píxel del frame original
    (antes de recortar), o None si no se detectó cara.

    Cada landmark: {"id", "x", "y", "z_mediapipe", "visibility"}
    - "id" es el índice original de MediaPipe (útil para saber qué
      punto del contorno es cada uno, ej. comisuras, centro del labio).
    - "z_mediapipe" es la profundidad relativa que estima el propio
      modelo (no es un valor real en mm; para eso se usa depthAligned).

    Importante: rgb_image debe estar en orden RGB (no BGR).
    """
    h, w = rgb_image.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        results = fm.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    points = []
    for idx in LIPS_IDX:
        lm = landmarks[idx]
        points.append({
           "id": idx,
            "x": lm.x * w,
            "y": lm.y * h,
            "z_mediapipe": lm.z,
            "visibility": getattr(lm, "visibility", None),
        })
    return points


def bbox_from_landmarks(points, img_w, img_h, margin_ratio=0.35):
    """Calcula el bbox cuadrado (con margen) a partir de los landmarks
    de labios ya detectados. Separado de detect_lips_landmarks para no
    tener que volver a correr FaceMesh si ya tienes los puntos."""
    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_min -= box_w * margin_ratio
    x_max += box_w * margin_ratio
    y_min -= box_h * margin_ratio
    y_max += box_h * margin_ratio

    return make_square_bbox(x_min, y_min, x_max - x_min, y_max - y_min, img_w, img_h)


def detect_lips_bbox(rgb_image, margin_ratio=0.35):
    """Devuelve (x, y, w, h) del bbox de los labios en coordenadas de la
    imagen RGB (con margen extra), o None si no se detectó cara.

    Se mantiene por compatibilidad con código existente (ej.
    batch_crop_mouths.py vía process_frame); internamente ahora
    reutiliza detect_lips_landmarks + bbox_from_landmarks.
    """
    h, w = rgb_image.shape[:2]
    points = detect_lips_landmarks(rgb_image)
    if points is None:
        return None
    return bbox_from_landmarks(points, w, h, margin_ratio=margin_ratio)


def landmarks_to_crop_coords(points, bbox, size=256):
    """Transforma landmarks de coordenadas del frame original a
    coordenadas del recorte size x size, usando el mismo bbox (cuadrado)
    que se usó en crop_resize. Conserva las demás propiedades de cada punto."""
    x0, y0, box_w, box_h = bbox
    scale = size / box_w  # bbox es cuadrado, box_w == box_h

    transformed = []
    for p in points:
        transformed.append({
            **p,
            "x": (p["x"] - x0) * scale,
            "y": (p["y"] - y0) * scale,
        })
    return transformed


def sample_depth_landmarks(depth_crop_m, points_crop_coords, scale=DEPTH_SCALE_MM):
    """Muestrea depth_crop (en metros, tal como lo entrega crop_resize
    sobre depthAligned) en cada landmark ya transformado al espacio del
    recorte, y devuelve la profundidad real en mm.

    Cada punto de salida: {"id", "x", "y", "depth_mm"}
    """
    h, w = depth_crop_m.shape[:2]
    depth_points = []
    for p in points_crop_coords:
        xi = int(np.clip(round(p["x"]), 0, w - 1))
        yi = int(np.clip(round(p["y"]), 0, h - 1))
        z = depth_crop_m[yi, xi]
        depth_mm = 0.0 if np.isnan(z) else float(z) * scale
        depth_points.append({
            "id": p["id"],
            "x": p["x"],
            "y": p["y"],
            "depth_mm": depth_mm,
        })
    return depth_points


def make_square_bbox(x, y, w, h, img_w, img_h):
    """Convierte un bbox rectangular en uno cuadrado (lado = max(w,h)),
    recentrado, y lo ajusta a los límites de la imagen."""
    side = max(w, h)
    cx, cy = x + w / 2, y + h / 2
    x0 = cx - side / 2
    y0 = cy - side / 2

    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))
    side = min(side, img_w - x0, img_h - y0)

    return int(round(x0)), int(round(y0)), int(round(side)), int(round(side))


def crop_resize(img, bbox, size=256, interpolation=None):
    x, y, w, h = bbox
    crop = img[y:y + h, x:x + w]
    if interpolation is None:
        interpolation = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def process_frame(rgb_image, depth_aligned, size=256, margin_ratio=0.35):
    """Pipeline completo para un frame ya alineado: devuelve
    (rgb_crop, depth_crop) o None si no se detectó boca.

    rgb_crop sigue en orden RGB (conviértelo con save_rgb_png al guardar).
    depth_crop sigue en metros (float, con NaN posibles).
    """
    bbox = detect_lips_bbox(rgb_image, margin_ratio=margin_ratio)
    if bbox is None:
        return None

    rgb_crop = crop_resize(rgb_image, bbox, size)
    # INTER_NEAREST para depth: evita mezclar valores válidos con NaN al interpolar
    depth_crop = crop_resize(depth_aligned, bbox, size, interpolation=cv2.INTER_NEAREST)
    return rgb_crop, depth_crop


def process_frame_with_landmarks(rgb_image, depth_aligned, size=256, margin_ratio=0.35):
    """Igual que process_frame, pero además calcula y devuelve TODOS los
    landmarks del mesh facial que caen dentro del recorte (no solo
    labios -- también barbilla, nariz baja, mejillas, lo que alcance a
    entrar en los 256x256), con su profundidad real muestreada del
    depth alineado.

    El recorte se sigue centrando con base en los labios (igual que
    antes); lo que cambia es qué landmarks se reportan al final.

    Devuelve (rgb_crop, depth_crop, rgb_landmarks, depth_landmarks) o
    None si no se detectó boca.

    - rgb_landmarks: [{"id","x","y","z_mediapipe","visibility"}, ...]
      en coordenadas de píxel del recorte (0-255).
    - depth_landmarks: [{"id","x","y","depth_mm"}, ...] mismas x,y que
      rgb_landmarks (para poder superponerlos), con la profundidad real.
    """
    h, w = rgb_image.shape[:2]
    all_points_full = detect_all_landmarks(rgb_image)
    if all_points_full is None:
        return None

    lips_points_full = [p for p in all_points_full if p["id"] in LIPS_IDX]
    bbox = bbox_from_landmarks(lips_points_full, w, h, margin_ratio=margin_ratio)

    rgb_crop = crop_resize(rgb_image, bbox, size)
    depth_crop = crop_resize(depth_aligned, bbox, size, interpolation=cv2.INTER_NEAREST)

    all_points_crop = landmarks_to_crop_coords(all_points_full, bbox, size=size)
    rgb_landmarks = landmarks_within_crop(all_points_crop, size=size)
    depth_landmarks = sample_depth_landmarks(depth_crop, rgb_landmarks)

    return rgb_crop, depth_crop, rgb_landmarks, depth_landmarks


def save_landmarks_txt(points, path):
    """Guarda una lista de landmarks (rgb o depth) como texto plano
    separado por comas: id,x,y[,depth_mm]"""
    with open(path, "w") as f:
        if "depth_mm" in points[0]:
            for p in points:
                f.write(f"{p['x']:.2f},{p['y']:.2f},{p['depth_mm']:.2f}\n")
        else:
            for p in points:
                f.write(f"{p['x']:.2f},{p['y']:.2f}\n")


def visualize_landmarks_overlay(rgb_crop, depth_crop_m, rgb_landmarks, depth_landmarks, path):
    """Dibuja los landmarks sobre el recorte RGB y sobre una versión
    normalizada (solo para visualizar) del recorte de profundidad,
    uno junto al otro -- útil para verificar que caen en la misma
    posición en ambas imágenes.

    rgb_crop: recorte en orden RGB (como lo entrega crop_resize).
    depth_crop_m: recorte de profundidad en metros (como lo entrega crop_resize).
    """
    bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR).copy()
    depth_mm = np.nan_to_num(depth_crop_m, nan=0.0) * DEPTH_SCALE_MM
    depth_norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = cv2.cvtColor(depth_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for p in rgb_landmarks:
        cv2.circle(bgr, (int(round(p["x"])), int(round(p["y"]))), 1, (0, 255, 0), -1)
    for p in depth_landmarks:
        cv2.circle(depth_vis, (int(round(p["x"])), int(round(p["y"]))), 1, (0, 255, 0), -1)

    cv2.imwrite(path, np.hstack([bgr, depth_vis]))


def save_rgb_png(rgb_crop, path):
    """Guarda un recorte RGB (orden RGB) como PNG normal.

    OpenCV siempre asume BGR al escribir, así que hay que convertir
    aquí -- y solo aquí, nunca antes de pasarle la imagen a MediaPipe.
    """
    bgr_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr_crop)


def save_depth_png16(depth_crop, path, scale=DEPTH_SCALE_MM):
    """Guarda depth (metros, float, puede tener NaN) como PNG de 16 bits
    sin signo, escalando a milímetros.

    - Los NaN (huecos sin dato) se guardan como 0, igual que el formato
      raw de Kinect.
    - Para recuperar metros al leer: depth_m = png16.astype(np.float32) / scale
    - Al leer con cv2.imread, usar cv2.IMREAD_UNCHANGED, si no OpenCV
      lo reduce a 8 bits por defecto.
    """
    depth_mm = np.nan_to_num(depth_crop, nan=0.0) * scale
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_mm)


def load_depth_png16(path, scale=DEPTH_SCALE_MM):
    """Lee un PNG de 16 bits guardado con save_depth_png16 y lo devuelve
    en metros (float32). Los píxeles en 0 quedan como 0.0 (sin dato)."""
    depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")
    return depth_mm.astype(np.float32) / scale


if __name__ == "__main__":
    rgb, depth_aligned = load_frame_from_mat("frame001.mat")
    result = process_frame_with_landmarks(rgb, depth_aligned, size=256)

    if result is None:
        print("No se detectó boca en ese frame.")
    else:
        rgb_crop, depth_crop, rgb_landmarks, depth_landmarks = result
        save_rgb_png(rgb_crop, "mouth_rgb_256.png")
        save_depth_png16(depth_crop, "mouth_depth_256.png")
        save_landmarks_txt(rgb_landmarks, "landmarks_rgb.txt")
        save_landmarks_txt(depth_landmarks, "landmarks_depth.txt")
        visualize_landmarks_overlay(rgb_crop, depth_crop, rgb_landmarks, depth_landmarks, "overlay_check.png")
        print(f"Landmarks de labios detectados: {len(rgb_landmarks)}")
        print("Recortes y landmarks guardados en la carpeta actual.")