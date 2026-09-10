"""

Código 7.
Recorre la carpeta de imágenes filtradas, genera la nube de puntos 3D usando Open3D y guarda los archivos .txt.

"""
import os
import glob
import cv2
import numpy as np
import open3d as o3d

# --- CONFIGURACIÓN DE LAS RUTAS RAÍZ ---
ROOT_FOLDERS = [
    r"D:\Mariana_Palacios\Fonemas\Fonema_A\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_BE\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_CHE\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_DA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_E\P1", 
    r"D:\Mariana_Palacios\Fonemas\Fonema_FA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_GA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_I\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_JA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_KA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_LA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_MA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_ÑA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_NO\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_O\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_PA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_RA\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_SI\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_TE\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_U\P1",
    r"D:\Mariana_Palacios\Fonemas\Fonema_YA\P1",
]
# --------------------------------------------

# Parámetros intrínsecos estándar de Kinect v2 (Profundidad)
WIDTH = 512
HEIGHT = 424
FX = 365.456  # Distancia focal X
FY = 365.456  # Distancia focal Y
CX = 256.0    # Centro óptico X
CY = 212.0    # Centro óptico Y

# Crear objeto de parámetros intrínsecos una sola vez para optimizar
INTRINSIC = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)

def main():
    print(f"=== INICIANDO PROCESAMIENTO MULTI-RAÍZ ===")
    
    for root_folder in ROOT_FOLDERS:
        if not os.path.exists(root_folder):
            print(f"\n[ERROR] La ruta raíz no existe y será omitida: {root_folder}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Buscando ensayos en: {root_folder}")
        print(f"{'='*50}")

        # Buscar todas las carpetas 'resultados_filtrados' en la raíz actual
        search_pattern = os.path.join(root_folder, "**", "resultados_filtrados")
        folders = glob.glob(search_pattern, recursive=True)
        
        print(f"Se encontraron {len(folders)} carpetas de resultados en esta raíz.")

        for input_folder in folders:
            # Definir la carpeta de salida un nivel arriba (en el ENSAYO)
            essay_folder = os.path.dirname(input_folder)
            output_folder = os.path.join(essay_folder, "nube_puntos")
            
            # Buscar las imágenes filtradas
            search_path = os.path.join(input_folder, "*_filtrada.png")
            depth_files = sorted(glob.glob(search_path))
            
            if not depth_files:
                print(f"  [OMITIDO] No hay imágenes filtradas en: {os.path.basename(essay_folder)}")
                continue

            os.makedirs(output_folder, exist_ok=True)
            print(f"\n  Procesando ensayo: {os.path.basename(essay_folder)}")
            print(f"   -> Origen: {input_folder}")
            print(f"   -> Destino: {output_folder}")
            
            n_ok = 0
            
            # Recorrer cada imagen usando tu lógica original
            for i, depth_path in enumerate(depth_files, start=1):
                base_name = os.path.splitext(os.path.basename(depth_path))[0]
                
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_img is None:
                    print(f"     [ERROR] No se pudo cargar: {os.path.basename(depth_path)}")
                    continue

                # Convertir a formato compatible con Open3D (Float32)
                depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))

                # Crear la nube de puntos
                pcd = o3d.geometry.PointCloud.create_from_depth_image(
                    depth_o3d, INTRINSIC, depth_scale=1000.0, depth_trunc=4.5
                )

                # Corregir la orientación (Kinect tiene el eje Y invertido)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                # Extraer las coordenadas XYZ como array de NumPy
                points = np.asarray(pcd.points)

                # Definir ruta y guardar con NumPy tal como lo hacía tu primer script
                output_filename = f"{base_name}_puntos.txt"
                output_path = os.path.join(output_folder, output_filename)
                
                np.savetxt(output_path, points, fmt="%.6f", delimiter=" ", comments="")
                n_ok += 1

                # Control visual en consola cada 50 archivos
                if i % 50 == 0:
                    print(f"    {i} / {len(depth_files)} nubes de puntos generadas...")

            print(f"   -> Finalizado: {n_ok} de {len(depth_files)} guardados correctamente.")

    print("\n=== PROCESAMIENTO MULTI-RAÍZ FINALIZADO ===")

if __name__ == "__main__":
    main()
