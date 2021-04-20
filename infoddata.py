'''
Sistema de clasificacion de las enfermedades de la piel 
por medio de imagenes

Autor: Jesser Lemus
'''
# Importamos los modulos y librerias necesarias
import os
import numpy as np
import pandas as pd
import re
from time import sleep

# Determinamos la direccion de las imagenes para brindar informacion de las categorias
# os.getcwd determina
path_skindeseases = os.path.join(os.getcwd(),'Dataset/skin_diseases') + os.sep

# Defimos un valor de cuenta, para que incremente cuando encuentre un directorio o una imagen
# Definimos un prev_root, como metodo de control al pasar de direccion de directorio
# dirs_name es un arreglo que guarda los nombres de los directorios
# dir_count es un arreglo que guarda la cuenta de las imagenes encontradas en un directorio
count = 0
prev_root = ''
dirs_name = []
dir_count = []

# 
print("LEYENDO IMAGENES....")
sleep(1)
for root, dirs, files in os.walk(path_skindeseases):
    for filename in files:
        if re.search("\.(jpg|png|jpeg)$", filename):
            count += 1
            if prev_root != root:
                prev_root = root
                dir_count.append(count)
                count = 0
    for name_dirs in dirs:
        dirs_name.append(name_dirs)
    
# Al ejecutar el for que recorre todos los directorios en busca de imaganes, 
dir_count.append(count)
dir_count = dir_count[1:]
dir_count[0] = dir_count[0] + 1

# Formato para imprimir la informacion en el formato adecuado
print("CATAGORIAS ENCONTRADOS", len(dirs_name))
print("=========================================================================================")
print('{:<70}{}'.format("NOMBRE DEL CATEGORIAS","NUMERO DE IMAGENES"))
print("=========================================================================================")
for i in range(len(dirs_name)):
    print('{:<70}{}'.format(dirs_name[i], dir_count[i]), sep='. ')

print("=========================================================================================")
print('{:<70}{}'.format("TOTAL DE IMAGENES ENCONTRADAS", sum(dir_count)),"\n")



