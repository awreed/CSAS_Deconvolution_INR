import torch
import numpy as np
from sim_csas_package.utils import normalize,  save_sas_plot, g2c
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from numpy.fft import fft2, ifft2
import lpips

class WienerDeconv:
  def __init__(self, device, circular):
    self.device = device
    self.circular = circular
    self.ind = None

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

  def recon(self, log_min, log_max, num_log_space, img, psf, gt_img, save_name):
    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    x_shape, y_shape = img.shape[0], img.shape[1]

    if psf.shape != img.shape:
      print("Wiener filter padding code might be wrong for non-square PSFs...")
      psf = psf.squeeze()
      img = img.squeeze()

      psf_size = psf.shape[0]

      pad_w = np.abs((psf_size - x_shape))
      pad_h = np.abs((psf_size - y_shape))
      psf = np.pad(psf, ((0, pad_w), (0, pad_h)), mode='constant')

    noise_suppresion_const = np.logspace(log_min, log_max, num_log_space)

    img = img / np.sqrt(self.energy(img))

    if self.circular:
      mask = np.zeros((x_shape, y_shape))
      _, ind = g2c(mask)
      self.ind = ind
      mask.ravel()[self.ind] = 1
      mask = mask.reshape(x_shape, y_shape)

    psf_new = np.zeros_like(psf, dtype=np.complex128)
    psf_new.real = psf.real
    psf_new.imag = psf.real

    psf_new = psf_new / np.sqrt(self.energy(psf_new))

    psnrs = []
    ssims = []
    perceps = []
    images = []

    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.device)

    for i, noise in enumerate(noise_suppresion_const):
      deconv_scene = self.wiener_filter2D(img, psf_new, noise)
      if self.circular:
        deconv_scene = mask*deconv_scene

      images.append(deconv_scene)

      if gt_img is not None:

        psnr_est = peak_signal_noise_ratio(normalize(gt_img),
                                           normalize(deconv_scene))
        ssim_est = structural_similarity(normalize(gt_img),
                                         normalize(deconv_scene))
        gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
                                                           ...].repeat(1, 3, 1, 1).to(self.device)
        est = torch.from_numpy(self.norm(deconv_scene.squeeze()))[None, None,
                                                                  ...].repeat(1, 3, 1, 1).to(self.device)
        percep = loss_fn_alex(gt, est).item()

        psnrs.append(psnr_est)
        ssims.append(ssim_est)
        perceps.append(percep)

        print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep:", percep)

      save_sas_plot(deconv_scene, os.path.join(save_name, "deconv_img_" + str(i) + '.png'))
      np.save(os.path.join(save_name, "deconv_img_" + str(i) + '.npy'), deconv_scene)

    return images, psnrs, ssims, perceps




      
