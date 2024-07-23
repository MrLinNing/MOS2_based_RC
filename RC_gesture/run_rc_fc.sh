#!/bin/bash

python -u main_rc.py --ARCHI "RC_readout" --GPU 1 2>&1 | tee -a ./logs/RC_FC_model.log
