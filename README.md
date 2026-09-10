# Code for Acquisition and Reading Phonemes and Visemes from the Mexican Spanish Dataset

The codes used to acquire and process the phonemes and visemes of Mexican Spanish have been published. This `README` file specifies the order in which the files provided in this repository should be executed.

The project uses **Python** and **MATLAB**.

## Dataset structure

The study focused on 21 phonemes of Mexican Spanish: A, BE, CHE, DA, E, FA, GA, I, JA, KA, LA, MA, ÑA, NO, O, PA, RA, SI, TE, U, and YA.

Each of these phonemes data was organized into folders named FONEMA_*, where the asterisk is replaced by the respective phoneme. In each of these folders, you will find subfolders corresponding to the 10 participants (P1, P2, …, P10). Within each participant’s folder are subfolders corresponding to the three repetitions that were recorded (labeled as ENSAYO1, ENSAYO2, and ENSAYO3). And within each of these “ENSAYO*” (ESSAY) subfolders, the following information is available:

- captura_3s_audio.wav: This is the audio file captured by the Kinect of the phoneme being pronounced. Each one weighs approximately 517 KB.
- landmarks: This is a folder containing two subfolders with landmarks extracted from RGB images (each file is 2.84 KB in size, and the entire folder is 245 KB) and depth images (each file is 4.24 KB in size, and the entire folder is 367 KB) of the image cropped to 256 x 256 pixels. The “x” and “y” coordinates range from 0 to 255, and the Z value in the depth image is in mm. The depth images use depth landmarks extracted from the image that has already been processed using the median filter after outliers were corrected.
- nube_puntos: This is a folder containing 90 .txt files with the “X,” “Y,” and “Z” coordinates of each of the captured frames, allowing them to be viewed in three dimensions; each file is 2.21 MB in size. It also contains a subfolder with the point clouds of the reference points obtained through reprojection. These landmarks were obtained by cropping the region of interest from the original image; each file is 4.74 KB in size. This folder takes up 200 MB of storage space.
- overlay: This is a folder containing 90 images in RGB and depth PNG formats (side by side) with landmarks overlaid to show what they look like and which landmarks are extracted from the cropped images using MediaPipe. Each of these images is _ in size, and each folder is  in size.

- recortes_boca: This is a folder containing 180 images in PNG format of the mouth region: 90 RGB images and 90 depth images. These images have been cropped to 256 x 256 pixels, because the RGB images were captured at 1920 x 1080 pixels and the depth images at 512 x 424 pixels; therefore, we cropped them to a size that can be processed by most neural networks architectures—in this case, 256 x 256 pixels. Depth images have been processed using a median filter to remove any outliers that may exist in the image. Each RGB image is 52 KB, each depth image is 14 KB, and each of these folders totals approximately 5.8 MB.

## Execution Order

The following is the execution order for the acquisition, processing, and cleaning of the dataset.

### MATLAB Code

#### 1. `capturar_sesion.m`

This is the **first file to execute**.

This script is used during the **data acquisition stage** and requires the **Kinect sensor to be connected**. The captured data are stored in a compressed `.mat` file.

#### 2. `descomprimiendo.m`

This is the **second file to execute**.

This script is used to decompress the previously captured data. The following files are obtained:

- RGB video.
- Depth video (8-bit).
- `.wav` audio files.
- RGB images corresponding to 90 video frames.
- Depth images corresponding to 90 video frames (16-bit).

#### 3. `align_frames_many_ejecutions.m`

This is the **third file to execute**.

This script is used to **align the RGB and depth images**, since they have different spatial resolutions:

- RGB: **1920 × 1080 pixels**
- Depth: **512 × 424 pixels**

The alignment allows the information from both sensors to correspond spatially.

---

### Python Code

#### 4. `batch_crop_mouths.py`

This script calls `crop_mouth_kinect.py` and is the **fourth file to execute**.

It is used to crop the previously aligned images. The following data are obtained:

- Cropped RGB images.
- Cropped depth images.
- Landmarks corresponding to the cropped region, stored in `.txt` files.

The crop size is **256 × 256 pixels** and focuses on the mouth region.

#### 5. `mediana_REPASO.py`

This is the **fifth file to execute**.

After cropping the images, some gaps or anomalous values may remain in the depth images. These values can exceed **1000 mm** and are treated as outliers.

To correct these anomalous values, a **median filter** is applied to the depth images.

#### 6. `nubepuntos_many_eject.py`

This is the **sixth file to execute**.

This script is used to generate the **3D point clouds** from the cropped mouth region.

#### 7. `nubepuntos_landmarks_corregido.py`

This is the **last file to execute**.

First, the landmarks are obtained from the complete image. The **region of interest (ROI)** corresponding to the **mouth and chin** is then extracted.

Finally, the remaining landmarks are projected into a **3D point cloud using reprojection**.
