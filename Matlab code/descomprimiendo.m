% Después de la captura de los datos (capturar_sesion.m), se realiza la 
% descompresión de los mismos. Se escribe la carpeta en el que está ubicado
% cada archivo .mat para que ahí se realice la descompresión y que se 
% generen los frames, videos y audio (llamando a la función 
% "procesar_sesion").
% Se pueden poner varios a la vez, pero, lo más recomendable es que
% el máximo 6 carpetas por ejecución, debido a que la laptop 
% usa mucha capacidad en CPU y puede llegar a sobrecalentarse.
% Luego va (align_frames_many_ejecutions.m)

sesiones = {'capturas\20260828_133551', };
for k = 1:numel(sesiones)
    procesar_sesion(sesiones{k});
end 