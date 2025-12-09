from PIL import Image
import numpy as np
import onnxruntime as ort
from urllib.request import urlopen
from io import BytesIO

print("Downloading image...", flush=True)
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
with urlopen(url) as response:
    img = Image.open(BytesIO(response.read())).convert("RGB").resize((200, 200))

print("Processing image...", flush=True)
x = np.array(img, dtype=np.float32)
x = x / 127.5 - 1
X = np.expand_dims(x, axis=0).transpose(0,3,1,2)
print("Input shape:", X.shape, flush=True)

print("Loading model...", flush=True)
session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Running model...", flush=True)
preds = session.run([output_name], {input_name: X})

print("Model output:", preds[0][0][0], flush=True)
