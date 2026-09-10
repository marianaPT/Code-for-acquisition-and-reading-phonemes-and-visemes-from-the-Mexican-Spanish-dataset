% Toma los datos crudos guardados por capturar_sesion.m y genera a partir 
% de ellos los videos y frames individuales de una sesión. Primero carga 
% datos_crudos.mat, y con el audio hace una normalización de amplitud 
% (lo amplifica al 98% del máximo permitido) antes de guardarlo como .wav. 
% Luego arma dos videos en MP4: uno de color a partir de los frames RGB, 
% y otro de profundidad en escala de grises, para el cual normaliza el 
% rango de profundidad (0 a 4500 mm) a 8 bits, porque el códec MP4 no 
% admite 16 bits directamente, y repite el canal gris tres veces para 
% simular RGB. Finalmente extrae también los frames individuales: los 
% de color como PNG normales, y los de profundidad como PNG de 16 bits 
% sin ningún ajuste, es decir, el dato crudo tal como lo entrega el sensor 
% en milímetros, que es el que después se usa en el resto del pipeline. 
% Al final imprime en consola dónde quedó guardado todo.
% 
% Este código es para la ejecución individual de una sesión, pero, puede ejecutarse 
% en bucle para varias sesiones a la vez desde descomprimiendo.m, que es el que 
% llama a esta función.


function procesar_sesion(carpetaSesion)
% Uso: procesar_sesion('capturas/20260810_143022')

load(fullfile(carpetaSesion, 'datos_crudos.mat'));

rangoProfundidad = [0 4500] / 65535;  % ajusta este rango si tu escena lo requiere

% 1. AUDIO
nombreAudio = fullfile(carpetaSesion, sprintf('captura_%ds_audio.wav', segundosGrabacion));
if valorMaximoAudio > 0
    audioAmplificado = datosAudio * (0.98 / valorMaximoAudio);
else
    audioAmplificado = datosAudio;
end
audiowrite(nombreAudio, audioAmplificado, frecuenciaMuestreo);
fprintf('-> Audio guardado como "%s"\n', nombreAudio);

% 2. VIDEO COLOR
nombreVideoColor = fullfile(carpetaSesion, sprintf('captura_%ds_color.mp4', segundosGrabacion));
vColor = VideoWriter(nombreVideoColor, 'MPEG-4');
vColor.FrameRate = fpsDeseado;
open(vColor);
for i = 1:numFrames
    writeVideo(vColor, colorImages(:,:,:,i));
end
close(vColor);
fprintf('-> Video Color guardado como "%s"\n', nombreVideoColor);

% 3.  VIDEO PROFUNDIDAD EN ESCALA DE GRISES -- 8-bit, obligatorio por el códec MP4
nombreVideoDepthGris = fullfile(carpetaSesion, sprintf('captura_%ds_profundidad_gris.mp4', segundosGrabacion));
vDepthGris = VideoWriter(nombreVideoDepthGris, 'MPEG-4');
vDepthGris.FrameRate = fpsDeseado;
open(vDepthGris);
for i = 1:numFrames
    frameNormalizado = imadjust(uint16(depthImages(:,:,i)), rangoProfundidad, [0 1]);
    frameGris8bit = im2uint8(frameNormalizado);
    writeVideo(vDepthGris, repmat(frameGris8bit, [1 1 3]));
end
close(vDepthGris);
fprintf('-> Video Profundidad en Grises guardado como "%s"\n', nombreVideoDepthGris);

% 4. FRAMES INDIVIDUALES
carpetaColor = fullfile(carpetaSesion, 'frames_color');
carpetaDepthCrudo = fullfile(carpetaSesion, 'frames_profundidad_16bit');        % dato real sin tocar (mm)
mkdir(carpetaColor);
mkdir(carpetaDepthCrudo);

for i = 1:numFrames
    % Color RGB normal
    imwrite(colorImages(:,:,:,i), fullfile(carpetaColor, sprintf('color_%04d.png', i)));

    frameCrudo = uint16(depthImages(:,:,i));

    % 1) Profundidad cruda en 16-bit, sin ningún ajuste -- el dato tal cual llega del sensor
    imwrite(frameCrudo, fullfile(carpetaDepthCrudo, sprintf('depth_raw_%04d.png', i)));

end

fprintf('-> %d frames guardados en "%s" y "%s"\n', numFrames, carpetaColor, carpetaDepthCrudo);

fprintf('-> Sesión "%s" procesada por completo.\n', carpetaSesion);

end