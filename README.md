# Toy MSA transformer
This example illustrates how to iteratively learn proein structures from sequences.

## Data
I generated the dataset using the following procedure:
1. Generate an alignment consisting of blocks (colored), each block is assigned to either pattern (alpha helix/beta sheet) or flexible fragments. Additionally in the beginning of each pattern the amino-acid encodes the position of this block relative to the previous one. Also each block goes to its own sequence number in the alignment. The first sequence does not contain any information about the protein structure:
![Alt Text](dataset/alignment.png)

2. Place the patterns according to the sequence:
![Alt Text](dataset/example.png)

3. Fit the full protein into the placed patterns, while keeping them rigid and fragments flexible. It looks like this (left: protein CA; right: RMSD):
![Alt Text](dataset/anim.gif)

Finally examples from the dataset look like:
![Alt Text](dataset/dataset.png)

To generate the data you need to create directories:
*dataset/train*, *dataset/test*. Then run the script *spatial_dataset.py* with appropriate parameters. 

## Training and testing
The script *main.py* is responsible for launching training and testing.


## Results
We trained the model on the dataset of 1000 examples, consisting of up to 8 blocks with each pattern size ranging from 5 to 10 aa, and 10 sequences in MSA.
The results are:

epoch | Train | Test
------| ----- | -----
0     | 6.90  | 8.36
100   | 1.98  | 2.98
