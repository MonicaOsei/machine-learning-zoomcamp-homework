FROM agrigorev/model-2025-hairstyle:v1
RUN pip install --no-cache-dir pillow numpy onnxruntime
COPY lambda_function.py .
CMD ["python3", "-u", "lambda_function.py"]
