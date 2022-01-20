#@title rudalle-sr setup
import os
import sys
import torch
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from rudalle.realesrgan.model import RealESRGAN


class ESRGANUpscale():
    def __init__(self,esrganscale):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"using {device} for esrganupscale")
        self.model = RealESRGAN(device, esrganscale)
        self.model.load_weights(f"models/RealESRGAN_x{esrganscale}.pth")
        print("Model loaded!")
    def gan_upscale(self,imgpath,outpath,return_image=False):
        input_image = Image.open(str(imgpath))     
        input_image = input_image.convert('RGB')
        with torch.no_grad():
            sr_image = self.model.predict(np.array(input_image))
        sr_image.save(outpath)
        if return_image:
            return sr_image
        else:
            return None

if __name__ == "__main__":
    esrgan = ESRGANUpscale(2)

    esrgan.gan_upscale("arty3.png","tmp.png")



