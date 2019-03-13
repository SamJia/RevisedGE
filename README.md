# RevisedGE

- This repository is the implementation of RevisedGE:
> On Geometric Structure of Activation Spaces in Neural Networks

## Files in the folder
- `data/`: sample datasets
- `RevisedGE.py`: source codes for RevisedGE

## Data format

RevisedGE can accept the data in skipgram format (e.g. data/Center.txt): first line is header (containing number of nodes and number of dimension), all other lines are node-id and d dimensional representation.

    2000    2
    0   -3.002371   -0.076243
    1   -3.002371   -0.005655
    2   -3.030305   0.437688
    3   -3.030305   -0.459201
    ...


## RevisedGE

#### Requirements
The code of RevisedGE has been tested running under Python 3.6.1, with the following packages installed (along with their dependencies):

- numpy == 1.12.1
- scipy == 1.1.0
- osqp == 0.4.1
- qpsolvers == 1.0.4


#### Basic usage
The usage of RevisedGE is as follow:

    python RevisedGE.py [-h] [-i INITIAL_SIZE] [-s CONVEXHULL_SIZE]
                    [-c CONVERGENCE_DISTANCE] [-e TOLERATED_QP_ERROR]
                    [-cr CONVERGENCE_CHANGE_RATE] [-si SIGMA]
                    input_filename output_filename

For example:

    python RevisedGE.py data/Center.npy results/Center_ach.txt
