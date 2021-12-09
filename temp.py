import glob
import os

image = "C39P4thinF_original_IMG_20150622_105335_cell_16.png"
if ".png" not in image:
    image += ".png"
image = glob.glob(os.path.join(
    os.getcwd(), 'Datos', '*', '*', image))[0]
print(image)
