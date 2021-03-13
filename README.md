# OpenFold2
![GitHub Logo](docs/Fig/OpenFold2_mid.png)

Attempt at reproduction of AlphaFold2. 
This repository is an ecxample of dataset-driven model development. First, we generate a dataset, using a procedure that mimics some aspect of the real data.
Then we develop a model, that tries to learn this particular dataset. 

## Toy datasets & models
1. [__toy_gpt__](https://github.com/lupoglaz/OpenFold2/tree/toy_gpt) : GPT model, mainly Karpathy's code but rewritten in a more structured way
2. [__toy_se3__](https://github.com/lupoglaz/OpenFold2/tree/toy_se3) : Iterative SE(3)-transformer and simple particle dynamics dataset
3. [__toy_prot__](https://github.com/lupoglaz/OpenFold2/tree/toy_prot): Toy protein dataset and structural part of the AlphaFold2 model
4. [__toy_msa__](https://github.com/lupoglaz/OpenFold2/tree/toy_msa): Toy multiple sequence alignment dataset, supervised case and complete MSA+structural parts of AlphaFold2
