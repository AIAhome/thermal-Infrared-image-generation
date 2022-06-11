import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        
        if mode == 'train' or mode == 'val':
            self.files_A = sorted(glob.glob(os.path.join(root, "images_rgb_%s" % mode, 'data') + "/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "images_thermal_%s" % mode, 'data') + "/*.*"))
        elif mode == 'test':
            self.files_A, self.files_B = self._get_map(root, 'rgb_to_thermal_vid_map.json')

            self.files_A = [os.path.join(root, "video_rgb_%s" % mode, 'data', self.files_A[i]) for i in range(len(self.files_A))]
            self.files_B = [os.path.join(root, "video_thermal_%s" % mode, 'data',self.files_B[i]) for i in range(len(self.files_B))]
        else:
            raise NotImplemented

    def _get_map(self, root, path):
        import json
        filepath = os.path.join(root, path)
        with open(filepath, 'r') as f:
            #data = f.read()
            data = json.load(f)
        files_A = list(data.keys())
        files_B = list(data.values())

        return files_A, files_B

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__=='__main__':
    dataset = ImageDataset(root='/data/FLIR_ADAS_v2', unaligned=False, mode='test')