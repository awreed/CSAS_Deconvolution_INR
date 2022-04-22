import os
import glob
import numpy as np
from render_parameters import RenderParameters
from inr_recon import INR_Recon
from grad_desc_recon import GradDescRecon
from bremen_alg import BremenAlg
from wiener_filter import WienerDeconv
from dip_recon import DIP_Recon
import cv2
from utils import imwrite, save_sas_plot
import matplotlib.pyplot as plt
from functools import partial
import pickle
from utils import crop_psf, save_plot, g2c, c2g


if __name__ == '__main__':
  DATA_DIR = '/home/awreed/SASDeconv/experiments/data_400_phase_grid_random'
  IMAGE_DIR = '/home/awreed/SASDeconv/experiments/images_400_phase_grid_random'
  SAVE_DIR = 'tmp3'
  PSF_SMALL = 'complex_psf_small.npy'
  PSF_BIG = 'complex_psf.npy'

  data_files = glob.glob(os.path.join(DATA_DIR, '*'))

  print("Detected", len(data_files), "data files")

  COMPUTE_PSF_SMALL = True

  NUM_IMAGES = 1
  NUM_NOISE = 7

  print("WARNING: Ensure render parameters match dataset creation")

  SIZE_SIM = 400
  SIZE_BF = 400
  dev = 'cuda:0'

  RP = RenderParameters(device=dev)
  RP.define_transducer_pos(theta_start=0, theta_stop=360,
      theta_step=1, r=0.85, z_TX=0.25, z_RX=0.25)
  RP.define_scene_dimensions(scene_dim_x=[-.4, .4],
      scene_dim_y=[-.4, .4],
      scene_dim_z=[0, 0],
      pix_dim_sim=[SIZE_SIM, SIZE_SIM, 1],
      pix_dim_bf=[SIZE_BF, SIZE_BF, 1],
      perturb=False, 
      circle=True)
  RP.generate_transmit_signal(crop_wfm=True)

  psf_big = np.load(os.path.join('../psf_data', 'psf0.npy'))
  psf_big = psf_big / np.sqrt(np.sum(np.abs(psf_big)**2))

  if COMPUTE_PSF_SMALL:
    #print("psf before", psf_big.shape)
    psf_small = crop_psf(RP, psf_big, thresh=0.96)
    #psf_small = psf_big
    np.save(os.path.join(DATA_DIR, PSF_SMALL), psf_small)
    #print("psf after", psf_small.shape)

  save_sas_plot(np.abs(psf_small), os.path.join(SAVE_DIR, 'small_psf.png'))

  psf_small, ind = g2c(psf_small)
  save_sas_plot(np.abs(psf_small), os.path.join(SAVE_DIR, 'small_psf_masked.png'))

  INR = INR_Recon(RP)
  GD = GradDescRecon(RP)
  BR = BremenAlg(RP)
  DIP = DIP_Recon(RP)
  Wiener = WienerDeconv(RP)

  methods = []
  sigma = 12
  MAX_LEN = 10
  save_every = 100
  max_iter = 5000
  sim=True
  save_path=''
  crop_size=None

  # INR
  meth = { 'name':'INR', 
           'func':partial(INR.recon, sigma, MAX_LEN, max_iter, 'none', 0.0,
                save_every, sim, psf_small, crop_size, 1e-3),
           'psnr':[[] for i in range(NUM_IMAGES)],
           'ssim':[[] for i in range(NUM_IMAGES)],
           'lpips':[[] for i in range(NUM_IMAGES)]
  }

  methods.append(meth)

  

  meth = { 'name':'INR_TV', 
            'func':partial(INR.recon, sigma, MAX_LEN, max_iter, 'tv', 1e-7,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  meth = { 'name':'INR_Grad_Reg', 
            'func':partial(INR.recon, sigma, MAX_LEN, 'grad_reg', 1e-10,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

 
  meth = { 'name':'INR_L1', 
            'func':partial(INR.recon, sigma, MAX_LEN, 'l1', 1e-9,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips': [[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  meth = { 'name':'INR_L2', 
            'func':partial(INR.recon, sigma, MAX_LEN, 'l2', 1e-8,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  # GD
  
  meth = { 'name':'GD', 
           'func':partial(GD.recon, MAX_LEN, max_iter, 'none', 0.0,
                save_every, sim, psf_small, crop_size),
           'psnr':[[] for i in range(NUM_IMAGES)],
           'ssim':[[] for i in range(NUM_IMAGES)],
           'lpips':[[] for i in range(NUM_IMAGES)]
  }

  #methods.append(meth)

  meth = { 'name':'GD_TV', 
            'func':partial(GD.recon, MAX_LEN, max_iter, 'tv', 1e-7,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  meth = { 'name':'GD_Grad_Reg', 
            'func':partial(GD.recon, MAX_LEN, max_iter, 'grad_reg', 1e-7,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

 #methods.append(meth)

 
  meth = { 'name':'GD_L1', 
            'func':partial(GD.recon, MAX_LEN, max_iter, 'l1', 1e-8,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  meth = { 'name':'GD_L2', 
            'func':partial(GD.recon, MAX_LEN, max_iter, 'l2', 1e-7,
                 save_every, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  # DIP
  
  meth = { 'name':'DIP', 
            'func':partial(DIP.recon, MAX_LEN, max_iter, 'none', 0.00,
                 save_every,sim,  psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  # Wiener
  
  meth = { 'name':'Wiener', 
            'func':partial(Wiener.recon, MAX_LEN, sim, psf_big, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim':[[] for i in range(NUM_IMAGES)],
            'lpips':[[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)


  # Bremen
  
  meth = { 'name':'Bremen', 
            'func':partial(BR.recon, MAX_LEN, 500, sim, psf_small, crop_size),
            'psnr':[[] for i in range(NUM_IMAGES)],
            'ssim': [[] for i in range(NUM_IMAGES)],
            'lpips': [[] for i in range(NUM_IMAGES)]
    }

  #methods.append(meth)

  

  # over all images
  for i in range(0, NUM_IMAGES):

    #if i == 0:
    #  continue

    # Load the ground truth image
    gt_name = os.path.join(IMAGE_DIR, 'gt' + str(i) + '.npy')
    gt_img = np.load(gt_name)
    gt_img = c2g(gt_img, RP.circle_indeces, SIZE_SIM)
    save_sas_plot(gt_img, os.path.join(SAVE_DIR, 'gt_img_orig.png'))

    if crop_size:
      gt_img = gt_img[crop_size//2:-crop_size//2, crop_size//2:-crop_size//2]

    save_sas_plot(gt_img, os.path.join(SAVE_DIR, 'gt_img_cropped.png'))

    gt_img, _ = g2c(gt_img)

    save_sas_plot(gt_img, os.path.join(SAVE_DIR, 'gt_img' + str(i) + '.png'))
    save_sas_plot(gt_img, os.path.join(SAVE_DIR, 'gt_img_mpl' + str(i) + '.png'))

    #if i < NUM_IMAGES - 1:
    #  continue

    # over all noise values
    for j in range(0, NUM_NOISE, 1):
      
      print("Processing Image", i + 1, "of", NUM_IMAGES,  "Noise level", j + 1, "of", NUM_NOISE)

      # Load the noise image
      image_name = os.path.join(DATA_DIR, 'beamformed_scatterers_' + str(i)
      + 'noise_' + str(j) + '.npy')

      image = np.load(image_name)

      image_grid = c2g(image, RP.circle_indeces, SIZE_SIM)

      #ig = image_grid[crop_size//2:-crop_size//2,
      #    crop_size//2:-crop_size//2]
      #ig_gt, ind = g2c(ig)

      save_sas_plot(image_grid.real, os.path.join(SAVE_DIR, 'gt_real.png'))
      save_sas_plot(image_grid.imag, os.path.join(SAVE_DIR, 'gt_imag.png'))


      save_sas_plot(np.absolute(image_grid), os.path.join(SAVE_DIR, 'BF' + str(i)
        + '_noise' + str(j) + '.png'))
      save_sas_plot(image_grid.real, os.path.join(SAVE_DIR, 'BF' + str(i)
        + '_noise_real' + str(j) + '.png'))
      save_sas_plot(image_grid.imag, os.path.join(SAVE_DIR, 'BF' + str(i)
        + '_noise_imag' + str(j) + '.png'))
      save_sas_plot(np.arctan2(image_grid.imag, image_grid.real), os.path.join(SAVE_DIR, 'BF' + str(i)
        + '_noise_phase' + str(j) + '.png'))


      # over all methods
      for meth in methods:
        print("Using", meth['name'])
        psnr, ssim, lpip, scene = meth['func'](gt_img, image, SAVE_DIR)
        meth['psnr'][i].append(psnr)
        meth['ssim'][i].append(ssim)
        meth['lpips'][i].append(lpip)

        print("PSNR", psnr, "SSIM", ssim, 'LPIPS', lpip)

        save_sas_plot(scene, os.path.join(SAVE_DIR, 'deconv_' + meth['name'] + '_image_' + str(i)
        + '_noise_' + str(j) + '.png'))

        np.save(os.path.join(SAVE_DIR, 'deconv_' + meth['name'] + '_image_'
          + str(i) + '_noise_' + str(j) + '.npy'), scene)


        path = os.path.join(SAVE_DIR, meth['name'] + 'image_' + str(i) + '_noise_' + str(j) + '.pickle')
        with open(path, 'wb') as handle:
          pickle.dump(meth, handle)


    plt.figure()
    for meth in methods:
      plt.plot(meth['psnr'][i], label=meth['name'])
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'Image_' + str(i) + 'PSNR.png'))


    plt.figure()
    for meth in methods:
      plt.plot(meth['ssim'][i], label=meth['name'])


    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'Image_' + str(i) + 'SSIM.png'))


    plt.figure()
    for meth in methods:
      plt.plot(meth['lpips'][i], label=meth['name'])

    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'Image_' + str(i) + 'LPIPS.png'))









    

      
