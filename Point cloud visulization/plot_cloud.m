% Este script de MATLAB carga un archivo de texto que contiene coordenadas 
% 3D de una nube de puntos y las visualiza en un gráfico 3D.

% 1. Cargar los datos del archivo de texto
datos = load("C:\Users\mptxd\Downloads\opencv\ENSAYO2\nube_puntos_boca\landmarks\color_0050_landmarks_boca.txt");

% 2. Separar las columnas en x, y y z
x = datos(:, 1);
y = datos(:, 2);
z = datos(:, 3);

% 3. Graficar los puntos en 3D
% Para graficar se cambio el orden de los ejes para que correspondiera
% a como estan los ejes de la camara del Kinect 2
 scatter3(x, z, y, 1,z,'filled');

grid on;
title('Nube de puntos 3D');
% Ojo se cambio las etiquetas de los ejes
xlabel('Eje X');
ylabel('Eje Z');
zlabel('Eje Y');
colorbar;