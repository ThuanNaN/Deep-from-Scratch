## ViT from Huggingface

### Feature extraction only
```python
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
```

### Runs
#### 1. Build and start triton container
```bash
make cont_up
```

#### 2. Run inference
```python
pip install tritonclient

python client.py
```
