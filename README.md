# Keras-Template
requres python >= 3.10
## 1. Create new env

### create new env using venv

```{bash}
python -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade && python -m pip install --upgrade pip
# or
python3 -m venv .venv
source .venv/bin/activate
python3 -m ensurepip --upgrade && python3 -m pip install --upgrade pip
```

## 2. Install packages

### option 1:  tensorflow
```{bash}
# tensorflow >=2.16
pip install tensorflow[and-cuda]==2.16.*

#keras 3
pip install keras==3.*
```
### option 2:  pytorch
```{bash}
# pytorch windows
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# pytorch linux or mac
pip3 install torch torchvision torchaudio

#keras 3
pip install keras==3.*
```

## 3. Install packages

```{bash}
pip install -r requirements.txt
```

## Deploy model

### [Convert to ONNX](https://github.com/onnx/tensorflow-onnx)

```{bash}
    pip install -U tf2onnx  
    python -m tf2onnx.convert --saved-model logs/saved_model --opset 14 --output logs/model.onnx
```
