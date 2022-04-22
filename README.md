# CSAS Deconvolution

## Requirements
1. Python >= 3.6
2. Conda
## Installation
1. Update conda using `conda update -n base conda`. 
2. Install the python virtual environment containing the necessary packages using `conda env create -f environ.yml`.
3. After installing, enter the virtual environment with `conda activate CSAS_INR_Deconv`.
4. Finally, install PyTorch using from command line like `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`. 
Visit `https://pytorch.org/` for more details. 

## Running the Deconvolution Simulation Pipeline (Coming 4/29/22)
The `example_sim_deconv_pipeline/deconvolve_simulated_scene.py` script creates two sets of simulated CSAS measurements,
reconstructs the measurements using DAS, and then deconvolves the images using our INR approach and baselines discussed 
in the paper. The simulation parameters are easily modified by editing the `example_sim_deconv_pipeline/sim_config.ini` 
file.

## Beamforming the AirSAS data (Coming 4/29/22)

## Deconvolving the simulated data (Coming 4/29/22)

## Deconvolving the AirSAS data (Coming 4/29/22)