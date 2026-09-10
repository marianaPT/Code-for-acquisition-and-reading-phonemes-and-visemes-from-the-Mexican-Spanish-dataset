% Este código se encarga de capturar audio, color y profundidad con el 
% Kinect al mismo tiempo. Antes de grabar, libera objetos de video que 
% hayan quedado abiertos de una ejecución anterior, crea una carpeta con 
% la fecha y hora de la sesión, y arma las cámaras con trigger manual, 
% dejando pasar 1.5 segundos para que el sensor de color se estabilice 
% antes de empezar a contar frames. También busca el micrófono por nombre 
% en vez de por ID fijo, porque el Kinect puede reordenar los dispositivos 
% de audio al encenderse. Al final dispara la grabación de audio y video 
% juntos, revisa que el audio no haya salido en silencio o muy bajo, y 
% guarda todo sin procesar en un .mat para que la captura no tome mucho 
% tiempo, y estos a su vez se guardan en una carpeta con la fecha y hora 
% de captura (Ejemplo: 20260828_133551). 
% Se le puede indicar a la persona en qué momento habría de hablar 
% (cuando en la consola diga "Grabando AUDIO, COLOR y PROFUNDIDAD") para
% que sea mucho más práctico. 
% La carpeta "capturas" debe estar guardada en la misma carpeta que este script (capturar_sesion.m) 
% y el resto de scripts.


% Liberar cualquier objeto que haya quedado abierto de una ejecución anterior
if exist('vColor','var'), try close(vColor); catch, end, end
if exist('vDepth','var'), try close(vDepth); catch, end, end
clear vColor vDepth colorVid depthVid grabadorM
imaqreset

% --- CREAR CARPETA ÚNICA PARA ESTA SESIÓN ---
marcaTiempo = datestr(now, 'yyyymmdd_HHMMSS');
carpetaSesion = fullfile('capturas', marcaTiempo);
mkdir(carpetaSesion);
fprintf('-> Carpeta de sesión: %s\n', carpetaSesion);

% --- CONFIGURACIÓN DE VIDEO ---
colorVid = videoinput('kinect', 1);
depthVid = videoinput('kinect', 2);

% Trigger manual: los objetos quedan "armados" con start() pero NO
% empiezan a contar frames hasta que se llame trigger(). Esto permite
% dejar que el sensor de color se estabilice (auto-exposure/balance de
% blancos) sin gastar frames de nuestro FramesPerTrigger.
triggerconfig(colorVid, 'manual');
triggerconfig(depthVid, 'manual');

fpsDeseado = 30;
segundosGrabacion = 3;
numFrames = fpsDeseado * segundosGrabacion;
colorVid.FramesPerTrigger = numFrames;
depthVid.FramesPerTrigger = numFrames;
colorVid.Timeout = segundosGrabacion + 5;
depthVid.Timeout = segundosGrabacion + 5;

% --- CONFIGURACIÓN DE AUDIO ---
frecuenciaMuestreo = 44100;
bitsPorMuestra = 16;
canalesAudio = 2;

%% --- ARMAR CÁMARAS Y ESPERAR A QUE SE ESTABILICE EL SENSOR ---
% Importante: el SDK del Kinect puede re-enumerar los dispositivos de
% audio del sistema al activarse, así que los IDs de audiodevinfo NO son
% fijos entre "antes" y "después" de encender las cámaras. Por eso
% buscamos el micrófono por NOMBRE, después de start(), en vez de usar
% un número de ID fijo.
start([colorVid, depthVid]);   % arma los objetos (con trigger manual, aún no cuenta frames)
pause(1.5);                    % tiempo de calentamiento para el auto-exposure del color

% --- Localizar el micrófono ---
info = audiodevinfo;
idDispositivoAudio = -1;      % respaldo: predeterminado del sistema si no se encuentra por nombre
nombreEncontrado = '(predeterminado del sistema)';
for k = 1:length(info.input)
    if contains(info.input(k).Name, 'Realtek', 'IgnoreCase', true)
        idDispositivoAudio = info.input(k).ID;
        nombreEncontrado = info.input(k).Name;
        break;
    end
end
fprintf('-> Usando dispositivo de audio ID %d: %s\n', idDispositivoAudio, nombreEncontrado);
grabador = audiorecorder(frecuenciaMuestreo, bitsPorMuestra, canalesAudio, idDispositivoAudio);

%% --- ARRANCAR TODO A LA VEZ: trigger de video + audio ---
trigger([colorVid, depthVid]);   % AHORA sí empieza a contar los numFrames reales
record(grabador);
fprintf('Grabando AUDIO, COLOR y PROFUNDIDAD por %d segundos...\n', segundosGrabacion);
wait(colorVid, segundosGrabacion + 5);
wait(depthVid, segundosGrabacion + 5);
stop(grabador);
fprintf('¡Grabación finalizada!\n');

%% --- EXTRACCIÓN ---
[colorImages, ~] = getdata(colorVid, numFrames);
[depthImages, ~] = getdata(depthVid, numFrames);
stop([colorVid, depthVid]);
delete([colorVid, depthVid]);

datosAudioCompleto = getaudiodata(grabador);
valorMaximoAudio = max(abs(datosAudioCompleto(:)));
fprintf('-> Amplitud máxima detectada: %.4f\n', valorMaximoAudio);
if valorMaximoAudio == 0
    warning('El micrófono grabó silencio absoluto. Verifica permisos de Windows.');
elseif valorMaximoAudio < 0.05
    warning('Amplitud muy baja (%.4f). Revisa el volumen/boost del micrófono en Windows.', valorMaximoAudio);
end

muestrasExactas = frecuenciaMuestreo * segundosGrabacion;
if length(datosAudioCompleto) >= muestrasExactas
    datosAudio = datosAudioCompleto(1:muestrasExactas, :);
else
    datosAudio = datosAudioCompleto;
end

%% --- GUARDAR SOLO LOS DATOS CRUDOS (rápido, sin procesar nada) ---
nombreMat = fullfile(carpetaSesion, 'datos_crudos.mat');
save(nombreMat, 'colorImages', 'depthImages', 'datosAudio', ...
    'valorMaximoAudio', 'fpsDeseado', 'segundosGrabacion', ...
    'frecuenciaMuestreo', 'numFrames', '-v7.3');
fprintf('-> Datos crudos guardados en "%s"\n', nombreMat);
fprintf('-> Sesión lista. Corre procesar_sesion(''%s'') cuando quieras generar videos/frames.\n', carpetaSesion);