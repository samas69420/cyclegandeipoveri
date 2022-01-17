basically the same code from the paper (https://arxiv.org/pdf/1703.10593.pdf) but simplified af



directory structure

cyclegandeipoveri/
|
|__ testB.jpg
|
|__ datasets/
|     |
|     |__ mydataset/
|           |
|           |__ trainA/
|           |     |
|           |     |__im0.jpg
|           |     ...
|           |     |__imN.jpg
|           |
|           |__ trainB/
|                 |
|                 |__im0.jpg
|                 ...
|                 |__imM.jpg
|__ salvataggi/
|       |
|       |__ pics/
|             |
|             |__ progressive/
|
|__ src/
     |
     |__ confg.py
     |
     |__ dataset.py
     |
     |__ main.py
     |
     |__ models.py
     |
     |__ nets.py
     |
     |__ utils.py
