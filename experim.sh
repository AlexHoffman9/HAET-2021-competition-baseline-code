#!/bin/bash
python training.py --net_sav /data/ahoffman/checkpoints/haet/resnet50_batch512lr.1.pt --lr .1
python training.py --net_sav /data/ahoffman/checkpoints/haet/resnet50_batch512lr.05.pt --lr .05
python training.py --net_sav /data/ahoffman/checkpoints/haet/resnet50_batch512lr.2.pt --lr .01
python training.py --net_sav /data/ahoffman/checkpoints/haet/resnet50_batch512lr.4.pt --lr .4

# python test.py --net_sav /data/ahoffman/checkpoints/haet/resnet18_batch512lr.1.pt 
# python test.py --net_sav /data/ahoffman/checkpoints/haet/resnet18_batch512lr.05.pt
# python test.py --net_sav /data/ahoffman/checkpoints/haet/resnet18_batch512lr.2.pt 
# python test.py --net_sav /data/ahoffman/checkpoints/haet/resnet18_batch512lr.4.pt 