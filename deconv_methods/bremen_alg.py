import torch
import numpy as np
from sim_csas_package.utils import normalize, save_sas_plot, g2c
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import collections
import lpips


class BremenAlg:
  def __init__(self, device, circular):
    self.device = device
    self.circular = circular

  # [-1, 1] normalization
  def norm(self, x):
    return 2*((x - x.min())/(x.max() - x.min())) - 1

  # Energy of signal
  def energy(self, x):
    if torch.is_tensor(x):
      x = x.detach().cpu().numpy()
    x = np.sum(np.absolute(x)**2)

    return x

  # Cross energy of signals
  def xenergy(self, a, b):
    if torch.is_tensor(a):
      assert torch.is_tensor(b) == True
      a = a.detach().cpu().numpy()
      b = b.detach().cpu().numpy()

    prod = np.sum(np.absolute(a*b))
    return prod

  def recon(self, max_iter, save_every, img, psf, gt_img, save_name):
    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    LPIPS_metric = True

    # Turn off LPIPs metric if dimenions are too small
    if img.shape[0] < 100 or img.shape[1] < 100:
      print("Dimensions too small to compute LPIPs metric. Turning LPIPs metric off. ")
      LPIPS_metric = False

    x_shape, y_shape = img.shape[0], img.shape[1]

    if self.circular:
      assert img.shape[0] == img.shape[1], "Image must be square (H == W) to do circular reconstruction."
      _, ind = g2c(img)
      self.ind = ind

    if self.circular:
      mask = torch.zeros((x_shape, y_shape)).to(self.device)
      mask.view(-1)[self.ind] = 1

    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.device)

    i = 1

    # Bremen considers only the real portion of the DAS recon.
    a0n = img.real
    # Normalize DAS reconstruction by the energy of signal
    a0n = a0n / np.sqrt(self.energy(a0n))
    a0n = torch.from_numpy(a0n).to(self.device)

    r = psf.real
    r = r / np.sqrt(self.energy(r))
    r = torch.from_numpy(r).to(self.device)

    l_i = a0n.clone()

    psnrs = []
    ssims = []
    perceps = []
    images = []

    count = 0

    for epoch in range(0, max_iter):
      # If circular then only use the elements within circle to compute energy.
      if self.circular:
        l_i.view(-1)[self.ind]\
          = l_i.view(-1)[self.ind]\
          / np.sqrt(self.energy(l_i.view(-1)[self.ind]))
      else:
        l_i = l_i / np.sqrt(self.energy(l_i))

      # current observation approximation
      a_i = torch.nn.functional.conv2d(l_i[None, None, ...], r[None, None,
        ...], padding='same').squeeze()

      if self.circular:
        a_i = a_i * mask

        a_i.view(-1)[self.ind] \
          = a_i.view(-1)[self.ind] \
            / np.sqrt(self.energy(a_i.view(-1)[self.ind]))
      else:
        a_i = a_i / np.sqrt(self.energy(a_i))

      # convergence factor
      p_i = self.xenergy(a0n, a_i)

      # observation correction term
      c_i = a0n - p_i*a_i

      # error weighted learning rate
      k = np.sqrt(self.energy(c_i))

      l_i = l_i + k*c_i

      i = i+1
      count = count + 1

      # If simulated data
      if epoch % save_every == 0 or epoch == max_iter - 1:
        deconv_scene = np.sqrt(l_i.detach().cpu().numpy()**2)

        images.append(normalize(deconv_scene))

        if gt_img is not None:
          psnr_est = peak_signal_noise_ratio(normalize(gt_img),
                                             normalize(deconv_scene))
          ssim_est = structural_similarity(normalize(gt_img),
                                           normalize(deconv_scene))

          if LPIPS_metric:
            gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
                                                               ...].repeat(1, 3, 1, 1).to(self.device)
            est = torch.from_numpy(self.norm(deconv_scene.squeeze()))[None, None,
                                                                      ...].repeat(1, 3, 1, 1).to(self.device)

            percep = loss_fn_alex(gt, est).item()
            perceps.append(percep)
          else:
            percep = "NaN"

          psnrs.append(psnr_est)
          ssims.append(ssim_est)

          print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep:", percep)

        save_sas_plot(deconv_scene, os.path.join(save_name, "deconv_img_" + str(epoch) + '.png'))
        np.save(os.path.join(save_name, "deconv_img_" + str(epoch) + '.npy'), deconv_scene)

    return images, psnrs, ssims, perceps

  

    
    
    
