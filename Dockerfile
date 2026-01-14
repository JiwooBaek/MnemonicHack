FROM tensorflow/tensorflow:2.16.1-gpu-jupyter

WORKDIR /tf

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
