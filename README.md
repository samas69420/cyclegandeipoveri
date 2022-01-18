kinda the same code from the paper (https://arxiv.org/pdf/1703.10593.pdf) but simplified af
     
## Directory Structure
    cyclegandeipoveri/
    ├─testB.jpg
    └─datasets/
        │  
        ├─mydataset/
        |     |
        |     ├─trainA/
        |     |    |
        |     |    ├─im0.jpg
        |     |    ...
        |     |    └─imN.jpg
        |     |    
        |     └─trainB/
        |          |
        |          ├─im0.jpg
        |          ...
        |          └─imM.jpg
        ├─salvataggi/
        |     └─pics/
        |         └─progressive/
        └─src/
           ├─config.py
           ├─dataset.py
           ├─main.py
           ├─models.py
           ├─utils.py
           └─nets.py
