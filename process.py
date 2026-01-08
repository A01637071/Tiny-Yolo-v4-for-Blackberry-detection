import glob
import os

# === CONFIGURA AQUÍ LAS RUTAS DE TUS CARPETAS ===
train_dir = '/content/darknet/data/images/train'
val_dir   = '/content/darknet/data/images/val'
test_dir  = '/content/darknet/data/images/test'

# === ARCHIVOS DE SALIDA ===
file_train = open('/content/darknet/data/train.txt', 'w')
file_val   = open('/content/darknet/data/val.txt', 'w')
file_test  = open('/content/darknet/data/test.txt', 'w')

# === EXTENSIONES A INCLUIR ===
exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

# === FUNCIÓN PARA ESCRIBIR IMÁGENES ===
def write_images(folder, outfile):
    for ext in exts:
        for path in glob.iglob(os.path.join(folder, ext)):
            outfile.write(path + "\n")

# === GENERAR LISTAS ===
write_images(train_dir, file_train)
write_images(val_dir, file_val)
write_images(test_dir, file_test)

file_train.close()
file_val.close()
file_test.close()

# === CONTADORES ===
def count_images(folder):
    count = 0
    for ext in exts:
        count += len(glob.glob(os.path.join(folder, ext)))
    return count

print("train.txt, val.txt y test.txt creados correctamente.")
print("Train:", count_images(train_dir))
print("Val:", count_images(val_dir))
print("Test:", count_images(test_dir))
