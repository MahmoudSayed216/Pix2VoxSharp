import os
import shutil

ds_path = "/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/Experimental/ShapeNetRendering/ShapeNetRendering"
classes = os.listdir(ds_path)

for cls in classes:
    path = os.path.join(ds_path, cls)
    files = os.listdir(path)

    for file in files:
        full_path = os.path.join(path, file)
        rendering_path = os.path.join(full_path, "rendering")
        images = os.listdir(rendering_path)
        i = 0
        for image in images:
            name = f'{i}.png'.rjust(6, '0')
            full_image_path = os.path.join(rendering_path, image)
            new_path = os.path.join(rendering_path, name)
            os.rename(full_image_path, new_path)
            i+=1
