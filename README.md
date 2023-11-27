This repository holds the anonymized source code for CVPR 2024 submission with ID 15957 (3D Facial Expressions through Analysis-by-Neural-Synthesis).

SMIRK is trained on CelebA, FFHQ, LRS3, and MEAD datasets. Make sure to download these, as well as the FLAME (https://flame.is.tue.mpg.de/) model.

To train a SMIRK model run

```bash
python train.py config.yaml
```
