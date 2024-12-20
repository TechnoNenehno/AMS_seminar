#!/bin/bash
python train.py -b LDR -n 1 -g 0 --round 25 --epochs 100 --batch 2 --output Bizjak_train -d datasets/ThoraxCBCT_dataset.json
python eval.py --gpu 0 --batch 1 --checkpoint weights/Bizjak_train

