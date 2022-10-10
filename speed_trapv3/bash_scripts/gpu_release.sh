#!/bin/sh
python - << import torch; 
python - << torch.cuda.empty_cache();
