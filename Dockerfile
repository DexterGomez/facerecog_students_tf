FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt-get update

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libqt5x11extras5 \
    libxkbcommon-x11-0

RUN pip install opencv-python scipy pillow matplotlib scikit-learn

WORKDIR /app