import torch
import numpy as np
from models import MLP_FF2D_MLP, FourierFeaturesVector
import collections
from utils import L1_reg, L2_reg, grad_reg, TV, save_img, normalize, imwrite, save_sas_plot, g2c, c2g
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from skimage import restoration
from numpy.fft import fft2, ifft2
import lpips


class WienerDeconv:
  def __init__(self, RP):
    self.RP = RP

  def norm(self, x):
    return 2*((x-x.min())/(x.max() - x.min())) - 1

  def energy(self, x):
    return np.sum(np.absolute(x)**2)

  
  def wiener_filter2D(self, img, kernel, K):
    #kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.fftshift(ifft2(dummy)))
    return dummy

  def wiener_filter(self, img, kernel, K):
    dummy = np.copy(img)
    dummy = np.fft.fft(dummy)
    kernel = np.fft.fft(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.fftshift(np.fft.ifft(dummy)))
    return dummy


  def recon(self, max_len, sim, psf, crop_size, gt_img, img, save_name):
    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    if crop_size:
      gt_img = gt_img[crop_size//2:-crop_size//2, crop_size//2:-crop_size//2]

    img_size = gt_img.shape[0]

    if psf.shape != img.shape:
      psf = psf.squeeze()
      img = img.squeeze()

      psf_size = psf.shape[0]
      #if psf_size % 2 == 0:
      #  raise Exception('check padding')
      

      pad_w = np.abs((psf_size - img_size))
      psf = np.pad(psf, ((0, pad_w), (0, pad_w)), mode='constant')

      psf = psf.ravel()[self.RP.circle_indeces]

    #snrs = np.linspace(0, .01, 100)
    snrs = np.logspace(-7, 0, 100)
    print(snrs)

    #scale = np.sqrt(1 / np.max(np.absolute(img)))
    #img = scale * img

    img = img / np.sqrt(self.energy(img))

    img = c2g(img, self.RP.circle_indeces, img_size)
    psf = c2g(psf, self.RP.circle_indeces, img_size)
    mask = np.zeros((img_size, img_size))
    mask.ravel()[self.RP.circle_indeces] = 1
    mask = mask.reshape(img_size, img_size)

    psf_new = np.zeros_like(psf, dtype=np.complex128)
    psf_new.real = psf.real
    psf_new.imag = psf.real

    psf_new = psf_new / np.sqrt(self.energy(psf_new))

    if sim:

      loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)

      psnrs = []
      ssims = []
      perceps = []
      images = []

      for i, snr in enumerate(snrs):
        #print(i)
        deconv = self.wiener_filter2D(img, psf_new, snr)
        deconv = mask*deconv

        if crop_size:
          deconv = deconv[crop_size//2:-crop_size//2, crop_size//2:
              -crop_size//2]

        psnr = peak_signal_noise_ratio(normalize(deconv), normalize(gt_img))
        ssim = structural_similarity(normalize(deconv), normalize(gt_img))

      

        gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
            ...].repeat(1, 3, 1, 1).to(self.RP.dev)
        est = torch.from_numpy(self.norm(deconv.squeeze()))[None, None,
            ...].repeat(1, 3, 1, 1).to(self.RP.dev)

        percep = loss_fn_alex(gt, est).item()

        psnrs.append(psnr)
        ssims.append(ssim)
        perceps.append(percep)
        images.append(deconv)

        #save_sas_plot(deconv, os.path.join(save_name, str(i) + '.png'))

        #print("PSNR:", psnr, "SSIM", ssim, "Percep", percep, "Noise", i, "/",
        #    len(snrs))

      val = max(psnrs)
      max_psnr = psnrs[psnrs.index(val)]
      max_ssim = ssims[psnrs.index(val)]
      max_percep = perceps[psnrs.index(val)]
      deconv = images[psnrs.index(val)]

      return max_psnr, max_ssim, max_percep, normalize(deconv)
    
    else:
      for i, snr in enumerate(snrs):
        deconv = self.wiener_filter2D(img, psf_new, snr)
        deconv = mask*deconv
        save_sas_plot(deconv, os.path.join(save_name, str(i) + '.png'))
        np.save(os.path.join(save_name, str(i) + '.npy'), deconv)

      return deconv




      
