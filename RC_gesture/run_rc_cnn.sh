#!/bin/bash

python -u main_rc.py --GPU 1 2>&1 | tee -a ./logs/RC_CNN_model.log
