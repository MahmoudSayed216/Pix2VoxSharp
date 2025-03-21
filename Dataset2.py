import os
import torch
import random
from torch.utils.data import Dataset
import json
from utils.utils import load_binvox_as_tensor
from PIL import Image
import torchvision.transforms as T
from utils.debugger import DEBUG


class ShapeNetDataset(Dataset):
    def __init__(self, dataset_path, json_file_path, split='train', transforms=None):
        self.dataset_path = dataset_path
        with open(json_file_path, 'r') as file: json_file = json.load(file)
        self.samples = []
        self.transforms = transforms
        n_clases = len(json_file)
        classes = [json_file[i]["taxonomy_id"] for i in range(n_clases)]
        for i in range(n_clases):
            examples = json_file[i][split]
            partial_paths = [os.path.join(classes[i], examples[j]) for j in range(len(examples))]
            self.samples.extend(partial_paths)

        self.length = len(self.samples)
        self.list_of_24 = [f'{i}.png'.rjust(6, '0') for i in range(24)]

    def __len__(self):
        return self.length
    
    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    
    def __getitem__(self, index):
        chosen_samples = random.sample(self.list_of_24, self.n_views_rendering)
        renders_dir = os.path.join(self.dataset_path, "ShapeNetRendering/ShapeNetRendering", self.samples[index], "rendering")
        volume_path = os.path.join(self.dataset_path, "ShapeNetVox32/ShapeNetVox32", self.samples[index], "model.binvox")
        renderings_full_path = [os.path.join(renders_dir, sample) for sample in chosen_samples]
        images = [Image.open(path).convert("RGB") for path in renderings_full_path]
        input_images = []
        for i in range(self.n_views_rendering):
            input_images.append(self.transforms(images[i]))

        images = torch.stack(input_images, dim=0)

        volume = load_binvox_as_tensor(volume_path)

        return images, volume
    
"/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/Experimental/ShapeNetRendering/ShapeNetRendering/02691156/1a9b552befd6306cc8f2d5fe7449af61/rendering/13.png"
"/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/Experimental/ShapeNetRendering/ShapeNetRendering/02691156/1a9b552befd6306cc8f2d5fe7449af61/rendering/13.png"