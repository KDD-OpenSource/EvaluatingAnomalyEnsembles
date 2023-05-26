Evaluates many different ensemble methods for anomaly detection.

Read the corresponding paper here: https://ls9-www.cs.tu-dortmund.de/publications/IJCNN2023.pdf

To implement your own ensemble method, add it to combinations.py, and use the combinations dictionary to specify which methods to evaluate.
Also make sure that the results folder exists: https://tu-dortmund.sciebo.de/s/tnCoUy9c6kknC18

Then execute main.py. As it has to evaluate very many ensembles, there is a (fairly simple) parallelisation build into main.py. For this call python3 main.py {index} up to modulo_max (currently 1500)
