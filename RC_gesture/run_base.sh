#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

python -u main_resnet.py 2>&1 | tee -a ./logs/resnet_model.log
