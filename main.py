import io
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, send_file
from flask_cors import CORS

sys.path.append('U-2-Net')
from u2net_portrait_demo import detect_single_face, crop_face, inference
from model import U2NET

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def run():
    data = request.files['data'].read()
    pil_img = Image.open(io.BytesIO(data))
    if pil_img.size[0] > 1024 or pil_img.size[1] > 1024:
        pil_img.thumbnail((1024, 1024))

    torch.cuda.empty_cache()
    cfg_net = app.config['U2N_NET']
    cfg_face_cascade = app.config['U2N_FACE_CASCADE']
    cv_img = pil_to_cv(pil_img)
    cv_face = detect_single_face(cfg_face_cascade, cv_img)
    cv_im_face = crop_face(cv_img, cv_face)
    cv_im_portrait = inference(cfg_net, cv_im_face)
    pil_result = cv_to_pil((cv_im_portrait * 255).astype(np.uint8))

    buf = io.BytesIO()
    pil_result.save(buf, 'PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


def pil_to_cv(img):
    new_img = np.array(img, dtype=np.uint8)
    if new_img.ndim == 2:
        pass
    elif new_img.shape[2] == 3:
        new_img = new_img[:, :, ::-1]
    elif new_img.shape[2] == 4:
        new_img = new_img[:, :, [2, 1, 0, 3]]
    return new_img


def cv_to_pil(img):
    new_img = img.copy()
    if new_img.ndim == 2:
        pass
    elif new_img.shape[2] == 3:
        new_img = new_img[:, :, ::-1]
    elif new_img.shape[2] == 4:
        new_img = new_img[:, :, [2, 1, 0, 3]]
    new_img = Image.fromarray(new_img)
    return new_img


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier(
        './U-2-Net/saved_models/face_detection_cv2/haarcascade_frontalface_default.xml')
    model_dir = './U-2-Net/saved_models/u2net_portrait/u2net_portrait.pth'
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
    else:
        net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    app.config['U2N_NET'] = net
    app.config['U2N_FACE_CASCADE'] = face_cascade

    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
