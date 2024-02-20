#!/bin/bash

if [ $# -eq 0 ]; then
  DATA_DIR="./"
else
  DATA_DIR="$1"
fi

# Install Python dependencies
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt

modelUrl = "https://drive.google.com/file/d/11yaV9vo7FeI5svtL_mcV0F_I8L39FZf9/view?usp=drive_link"

# Download TF Lite model
FILE=${DATA_DIR}/model.tflite
if [ ! -f "$FILE" ]; then
  curl \
    -L "$modelUrl" -o ${FILE}
fi