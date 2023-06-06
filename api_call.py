import json
import requests
import io
import base64
import cv2
from PIL import Image, PngImagePlugin
import numpy as np
import random 
import glob
import os

api_url = 'http://127.0.0.1:7860'

subset = 'urpm_simple_corrected'
#base_prompt = 'selfie of a $nationality_toggle $sex_toggle with realistic $skin_toggle skin $extras realistic $eye_color_toggle eyes with round iris and $hair_length_toggle $hair_type_toggle $hair_color_toggle hair looking at the camera, headshot, well illuminated, detailed background, realistic symmetrical eyes, round iris, natural light, realistic lips'
base_prompt = 'natural selfie of a $nationality_toggle $sex_toggle with natural textured $skin_toggle skin $extras detailed $eye_color_toggle eyes with round iris and $hair_length_toggle $hair_type_toggle $hair_color_toggle hair, well illuminated, headshot, detailed and sharp background, natural light'

negative_prompt = "child, childish, rendering, 3d rendering, naked, nude, porn, blurry, ugly, bad anatomy, deformed body, missing fingers, extra fingers, deformed face, cropped, cropped face, chibi, weird eyes, worst quality, low quality, watermark, text, multiple faces, two faces"

if glob.glob('data/SD1.5/'+subset+'/*.png'):
    last = np.max([int(os.path.basename(name).split('.')[0].split('_')[-1]) for name in glob.glob('data/SD1.5/*.png')])
else:
    last = 0

print(last)
batch_size=8

for index in range(1000):
    print(index+1,"/",1000)
    prompt = base_prompt
    prompt = prompt.replace('$sex_toggle', random.choice(['man', 'woman', 'male', 'female']))
    prompt = prompt.replace('$hair_length_toggle', random.choice(['long', 'short', 'medium', 'extremely long']))
    prompt = prompt.replace('$hair_type_toggle', '') if 'buzzcut' in prompt else prompt.replace('$hair_type_toggle', random.choice(['curly', 'wavy', 'straight', 'messy']))
    prompt = prompt.replace('$hair_color_toggle', random.choice(['blonde', 'black', 'ginger', 'brown']))
    prompt = prompt.replace('$eye_color_toggle', random.choice(['brown', 'amber', 'hazel', 'gray']))
    prompt = prompt.replace('$nationality_toggle', random.choice(['european', 'north-american', 'hispanic', 'middle-eastern', 'asian', 'african']))
    prompt = prompt.replace('$skin_toggle', random.choice(['pale', 'fair', 'olive', 'brown']))
    prompt = prompt.replace('$extras', random.choice(['']))

    payload = {
        "prompt":prompt,
        "negative_prompt":negative_prompt,
        "steps":70,
        "sampler_index":"UniPC",
        "width":512,
        "height":512,
        "cfg_scale":8,
        "batch_size":batch_size,
    }

    response = requests.post(url=f'{api_url}/sdapi/v1/txt2img', json=payload)
    #print(response)
    r = response.json()
    #print(r)

    for img_idx, img in enumerate(r['images']):
        data_raw = io.BytesIO(base64.b64decode(img.split(",",1)[0]))
        #nparr = np.asarray(bytearray(data_raw.read()), dtype=np.uint8)
        #img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #cv2.imshow("test", img_np)
        #cv2.waitKey(1)

        image = Image.open(data_raw)

        #png_payload = {
        #    "image": "data:image/png;base64," + i
        #}

        #response2 = requests.post(url=f'{api_url}/sdapi/v1/png-info', json=png_payload)

        #pnginfo = PngImagePlugin.PngInfo()
        #pnginfo.add_text("parameters", response2.json().get("info"))
        image.save('data/SD1.5/'+subset+'/output_'+str(last+(index*batch_size)+img_idx+1)+'.png')
