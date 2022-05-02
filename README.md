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

Running the pipeline will generate a results figure in each `deconv_dir/image*` directory showing the deconvolution 
results of all the methods run on the created dataset.

<img alt="Image" height="175" src="example_sim_deconv_pipeline\deconv_dir\image0\deconvolution_results.png" width="1200"/>

Additionally, one can find bar plots displaying deconvolution metrics for each method.

<img height="200" src="example_sim_deconv_pipeline\deconv_dir\image0\psnr_bar_plot.png" width="200"/>
<img height="200" src="example_sim_deconv_pipeline\deconv_dir\image0\ssim_bar_plot.png" width="200"/>
<img height="200" src="example_sim_deconv_pipeline\deconv_dir\image0\lpips_bar_plot.png" width="200"/>


## Deconvolving AirSAS data (Coming 5/6/22)
