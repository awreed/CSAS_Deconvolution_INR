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



## Deconvolving Simulated data 
The `example_sim_deconv_pipeline/deconvolve_simulated_scene.py` script creates two sets of simulated CSAS measurements,
reconstructs the measurements using DAS, and then deconvolves the images using our INR approach and baselines discussed 
in the paper. The simulation and geometry parameters are modified by editing using the 
`example_sim_deconv_pipeline/system_parameters.ini` and `example_sim_deconv_pipeline/simulation.ini`
files. The deconvolution parameters can be edited using the `example_sim_deconv_pipeline/deconv.ini`. All `.ini` files
are commented to provide instructions for use. The output 
of the deconvolution methods are saved in the `deconv_dir` directory. 


## Deconvolving AirSAS data (Coming 5/6/22)

