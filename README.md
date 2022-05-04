# CSAS Deconvolution

Code for SINR: Deconvolving Circular SAS Images Using Implicit Neural Representations found at https://arxiv.org/abs/2204.10428.

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


## Deconvolving AirSAS data

The script `airsas_deconv_pipeline/reconstruct_scene.py` will reconstruct an AirSAS scene, compute the PSF, and then 
deconvolve the scene using the methods specified in the associated deconv.ini file. For example, one can use the INR to deconvolve 
AirSAS small feature cutout scene shown in the paper figure:

<img height="200" src="airsas_deconv_pipeline\20k_scene\fig-a.PNG" width="200"/>. 

Running `airsas_deconv_pipeline/reconstruct_scene.py` will populate the directory `airsas_deconv_pipeline\20k_scene` 
with INR deconvolution results for the 20k small feature DAS scene.

Running the script shoould yield a DAS reconstruction that looks like this:

DAS Reconstruction (found at `airsas_deconv_pipeline\20k_scene\scene_abs.png`) :
<img height="200" src="airsas_deconv_pipeline\20k_scene\scene_abs.png" width="200"/>

and an INR Deconvolution (found at `airsas_deconv_pipeline\20k_scene\image0\INR\deconv_img_100.png`) at epoch 100 
that looks like this:
<img height="200" src="airsas_deconv_pipeline\20k_scene\image0\INR\deconv_img_100.png" width="200"/>
