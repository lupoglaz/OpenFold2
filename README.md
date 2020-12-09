# Toy SE3 transformer
This example illustrates how SE3-transformer works on a really simple dataset.

## Data
I used code: https://github.com/pmocz/nbody-python to generate traces for particles subject to gravitational pull. 
The transformer should predict positions and velocities of particles at step __t+50__ using positions and velocities from step __t__.
To generate this dataset use the script __dataset/generate_dataset.py__
