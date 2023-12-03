dependencies = [
    'torch', 'numpy', 'cv2', 'onnxruntime',
]

import os
import cv2
import torch
import numpy as np

import onnxruntime as ort

import logging

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("animegan")

def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8: # resize image to multiple of 8s
        def to_x8s(x):
            return 256 if x < 256 else x - x%8 # if using tiny model: x - x%16
        img = cv2.resize(img, (to_x8s(w), to_x8s(h)))
    return img/127.5 - 1.0

class AnimeGAN():

    def __init__(self, onnxFile, model, map_location=None):
        self.session = ort.InferenceSession(onnxFile)
        if map_location == torch.device('cpu'):
            self.session.set_providers(['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        for i in range(len(inputs)):
            log.debug("Input[{}]: name={}, shape={}, type={}".format(i, inputs[i].name, inputs[i].shape, inputs[i].type))
        outputs = self.session.get_outputs()
        for i in range(len(outputs)):
            log.debug("Output[{}]: name={}, shape={}, type={}".format(i, outputs[i].name, outputs[i].shape, outputs[i].type))
        self._model = model

    def __call__(self, image, **kwargs):
        
        if self.session is None:
            self.load()

        # preprocess
        scale = image.shape[:2]
        image = preprocessing(image)
        image = image.astype(float)
        
        # infer
        output = self.session.run(None, {self._model[1]: [image]})[0]
        output = output[0]
        
        # postprocess
        output = (output + 1.) / 2 * 255
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (scale[1], scale[0]))
        
        return output
        
def animegan(
    progress=True, map_location=None,
    id="v3hayao",
):

    MODELS = {
        "v3hayao":          ("https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx",       "AnimeGANv3_input:0"),
        "v3jpface":         ("https://huggingface.co/vumichien/AnimeGANv3_JP_face/resolve/main/AnimeGANv3_JP_face.onnx",               "AnimeGANv3_input:0"),
        "v3portraitsketch": ("https://huggingface.co/vumichien/AnimeGANv3_PortraitSketch/resolve/main/AnimeGANv3_PortraitSketch.onnx", "animeganv3_input:0"),
        "v2hayao":          ("https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",                   "generator_input:0"),
        "v2paprika":        ("https://huggingface.co/vumichien/AnimeGANv2_Paprika/resolve/main/AnimeGANv2_Paprika.onnx",               "generator_input:0"),
        "v2shinkai":        ("https://huggingface.co/vumichien/AnimeGANv2_Shinkai/resolve/main/AnimeGANv2_Shinkai.onnx",               "generator_input:0"),
    }

    model = MODELS[id]
    url = model[0]
    
    TARGET = os.path.join(torch.hub.get_dir(), "AnimeGAN")
    FILENAME = url.split("/")[-1]

    path = os.path.join(TARGET, FILENAME)
    os.makedirs(TARGET, exist_ok=True)
    if not os.path.isfile(path):
        print("Downloading {} to {}".format(url, path)) 
        torch.hub.download_url_to_file(url, path, progress=progress)
  
    return AnimeGAN(path, model, map_location)
