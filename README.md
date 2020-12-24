# U^2-Net HTTP

This is an HTTP service wrapper for [U^2\-Net: Going Deeper with Nested U\-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net).

This repository was inspired by [cyrildiagne's basnet-http repository](https://github.com/cyrildiagne/basnet-http) and referenced it.

Spedical thanks to [@cyrildiagne](https://github.com/cyrildiagne) !

## Run U^2-Net HTTP server
```
python main.py
```

## Post an image file
```bash
curl -F "data=@input.jpg" http://localhost:8080 -o output.png
```

---

## Development

### 1. Setup
This repository requires folliwing libraries.
- python 3.6.12
- pytorch 1.7.1
- torchvision 0.8.2
- numpy 1.16.0
- pillow 7.2.0
- scikit-image 0.17.2
- opencv-python 3.4.2.17
- Flask 1.1.1
- flask-cors 3.0.8
- gunicorn 19.9.0

Note: If you have conda, you can create conda environment by following steps.

Example for Windows + CPU
```
conda create -n u2net-http python=3.6
activate u2net-http
conda install pytorch torchvision cpuonly -c pytorch
pip install -r requirements.txt
```

Of course, this repository will also work on GPU and Mac/Linux environments. For more information on how to install PyTorch, please refer to [PyTorch official site](https://pytorch.org/get-started/locally/).

### 2. Clone repositories
- Clone this ```U-2-Net-http``` repository
   ```
   git clone https://github.com/adakoda/U-2-Net-http.git
   ```
- Move to cloned directory and clone additional [U^2-Net](https://github.com/adakoda/U-2-Net) repository
  ```
  cd U-2-Net-http
  ```
  ```
  git clone https://github.com/adakoda/U-2-Net.git'
  ```

### 3. Download pretrained model weight files
- Run model weights download script in U-2-Net-http/U-2-Net directory
  ```
  python setup_model_weights.py
  ```
  After finished python script, you will get these files.
  ```
  U-2-Net-http/U-2-Net/saved_models/u2net/u2net.pth ... 168 MB
  U-2-Net-http/U-2-Net/saved_models/u2net_portrait/u2net_portrait.pth ... 168 MB
  ```
