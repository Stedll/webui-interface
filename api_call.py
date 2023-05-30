import json
import requests
import io
import base64
import cv2
from PIL import Image, PngImagePlugin
import numpy as np
import random 

api_url = 'http://xxx.xxx.xxx.xxx:7860'

base_prompt = 'portrait photo of a $nationality_toggle $sex_toggle with $skin_toggle skin realistic $eye_color_toggle eyes and $hair_length_toggle $hair_type_toggle $hair_color_toggle hairs looking at the camera, centered image composition, mugshot, studio light, full face, detailed face, face pores, detailed skin, detailed background, half body shot, realistic symmetrical eyes, round iris, eyelashes, ultra quality, ultra detailed, 32mm, natural light, face pores, realistic lips, detailed background'
negative_prompt = "lowres, ugly, bad anatomy, deformed body, missing fingers, extra fingers, blurry, deformed face, cropped, cropped face, chibi, weird eyes, worst quality, low quality"


for index in range(10):
    prompt = base_prompt
    prompt = prompt.replace('$sex_toggle', random.choice(['man', 'woman', 'boy', 'girl', 'male', 'female']))
    prompt = prompt.replace('$hair_length_toggle', random.choice(['long', 'short', 'medium', 'extremely long', 'buzzcut']))
    prompt = prompt.replace('$hair_type_toggle', '') if 'buzzcut' in prompt else prompt.replace('$hair_type_toggle', random.choice(['curly', 'wavy', 'straight', 'messy', 'bob cut']))
    prompt = prompt.replace('$hair_color_toggle', random.choice(['blonde', 'black', 'ginger', 'brown']))
    prompt = prompt.replace('$eye_color_toggle', random.choice(['brown', 'hazel', 'honey like', 'grey']))
    #prompt = prompt.replace('$eye_color_toggle', random.choice(['blue', 'brown', 'hazel', 'honey like', 'grey']))
    prompt = prompt.replace('$nationality_toggle', random.choice(['european', 'american', 'middle-eastern', 'asian', 'african']))
    prompt = prompt.replace('$skin_toggle', random.choice(['fair', 'medium', 'olive', 'brown', 'pale']))

    payload = {
        "prompt":prompt,
        "negative_prompt":negative_prompt,
        "steps":50,
        "sampler_index":"UniPC",
        "width":512,
        "height":512,
    }

    response = requests.post(url=f'{api_url}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    for i in r['images']:
        data_raw = io.BytesIO(base64.b64decode(i.split(",",1)[0]))
        nparr = np.asarray(bytearray(data_raw.read()), dtype=np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow("test", img_np)
        cv2.waitKey(1)

        image = Image.open(data_raw)

        png_payload = {
            "image": "data:image/png;base64," + i
        }

        response2 = requests.post(url=f'{api_url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save('data/output_'+str(index)+'.png', pnginfo=pnginfo)
