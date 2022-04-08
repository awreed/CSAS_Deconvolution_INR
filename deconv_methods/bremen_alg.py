import torch
import numpy as np
from sim_csas_package.utils import normalize, save_sas_plot, c2g
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import collections
import lpips


class BremenAlg:
  def __init__(self, RP):
    self.RP = RP

  # [-1, 1] normalization
  def norm(self, x):
    return 2*((x - x.min())/(x.max() - x.min())) - 1

  # Energy of signal
  def energy(self, x):
    if torch.is_tensor(x):
      dev = x.device
      x = x.detach().cpu().numpy()
    x = np.sum(np.absolute(x)**2)

    return x

  # Cross energy of signals
  def xenergy(self, a, b):
    if torch.is_tensor(a):
      assert torch.is_tensor(b) == True
      dev = a.device
      a = a.detach().cpu().numpy()
      b = b.detach().cpu().numpy()


    prod = np.sum(np.absolute(a*b))
    return prod

  def bremen(self, img, psf, gt_img, max_iter, max_len, sim=True, save_name='',
      crop_size=None):
    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)

    if crop_size:
      gt_img = gt_img[crop_size//2:-crop_size//2, crop_size//2:-crop_size//2]

    i = 1
    a0n = img / np.sqrt(self.energy(img))
    a0n = a0n.real

    a0n = c2g(a0n, self.RP.circle_indeces, gt_img.shape[0])
    a0n = torch.from_numpy(a0n).to(self.RP.dev)

    #r = psf / np.sqrt(self.energy(psf))
    r = psf
    r = r.real

    l_i = a0n.clone()

    a0n = a0n.view(-1)[self.RP.circle_indeces]

    d = collections.deque(maxlen=max_len)
    psnr_opt = []
    ssim_opt = []
    percep_opt = []
    images = []

    count = 0

    r = torch.from_numpy(r).to(self.RP.dev)

    while True:
      print(count)
      # normalization
      l_i.view(-1)[self.RP.circle_indeces]\
        = l_i.view(-1)[self.RP.circle_indeces]\
        / np.sqrt(self.energy(l_i.view(-1)[self.RP.circle_indeces]))

      # current observation approximation
      a_i = torch.nn.functional.conv2d(l_i[None, None, ...], r[None, None,
        ...], padding='same').squeeze()

      a_i = a_i.view(-1)[self.RP.circle_indeces]
      #a_i = scipy.signal.convolve2d(l_i, r, mode='same')
      a_i = a_i / np.sqrt(self.energy(a_i))

      # convergence factor
      p_i = self.xenergy(a0n, a_i)
      #p_i = 1

      # observation correction term
      c_i = a0n - p_i*a_i

      # error weighted learning rate
      k = np.sqrt(self.energy(c_i))

      # update current estimate
      l_i.view(-1)[self.RP.circle_indeces]\
        = l_i.view(-1)[self.RP.circle_indeces] + k*c_i

      i = i+1
      count = count + 1

      # If simulated data
      if sim:
        
        scene = np.sqrt(l_i.detach().cpu().numpy()**2)
        images.append(scene)
        #scene = scene / scene.max()
        #scene = drc(scene, np.median(scene), 0.2)
        if crop_size:
          scene = scene[crop_size//2:-crop_size//2, crop_size//2:-crop_size//2]

        #imwrite(scene, 'plots_200/l_i_' + str(i) + '.png')
        #save_sas_plot(scene, os.path.join(save_name, 'l_i_' + str(i) + '.png'))
        psnr = peak_signal_noise_ratio(normalize(scene),\
            normalize(gt_img))
        ssim = structural_similarity(normalize(scene),\
            normalize(gt_img))

        # The shape is : [H, W] --> [1, 3, H, W]
        gt = torch.from_numpy(self.norm(gt_img.squeeze()))
        gt = gt[None, None, ...].repeat(1, 3, 1, 1).to(self.RP.dev)

        # [H, W] --> [1, 3, H, W]
        est = torch.from_numpy(self.norm(scene.squeeze()))
        est = est[None, None, ...].repeat(1, 3, 1, 1).to(self.RP.dev)
        
        percep = loss_fn_alex(gt, est).item()

        print("i", i, "PSNR", psnr, "SSIM", ssim, "Percep", percep)
        d.append(psnr)
        ssim_opt.append(ssim)
        psnr_opt.append(psnr)
        percep_opt.append(percep)

        if len(d) == max_len:
          if d[-1] < d[0] or count > max_iter:
            val = max(psnr_opt)
            max_psnr = psnr_opt[psnr_opt.index(val)]
            max_ssim = ssim_opt[psnr_opt.index(val)]
            max_percep = percep_opt[psnr_opt.index(val)]
            scene = images[psnr_opt.index(val)]
            break

      # If real data
      else:
        deconv_scene = np.sqrt(l_i.detach().cpu().numpy()**2)
        if count > max_len:
          return normalize(deconv_scene)
        else:
          if count % 100 == 0:
            save_sas_plot(deconv_scene, os.path.join(save_name, str(count) + '.png'))
            np.save(os.path.join(save_name, str(count) + '.npy'), deconv_scene)
          #save_sas_plot(deconv_scene, os.path.join(save_name, 'mpl_log'
          #  + str(count) + '.png'), log=True)

    
    return max_psnr, max_ssim, max_percep, scene

  
  def recon(self, max_len, max_iter, sim, psf, crop_size, gt_img, img, save_name):
    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    scale = np.sqrt(1 / np.max(np.absolute(img)))
    img = scale * img

    #print(img.shape, psf.shape)
    if sim:
      psnr, ssim, percep, deconv = self.bremen(img, psf, gt_img,
          max_iter=max_iter, max_len=max_len, sim=sim, save_name=save_name,
          crop_size=crop_size)
      deconv = np.absolute(deconv)
      return psnr, ssim, percep, deconv


    else:
      deconv = self.bremen(img, psf, gt_img,
          max_iter=max_iter, max_len=max_len, sim=sim, save_name=save_name)
      deconv = np.absolute(deconv)
      return deconv



    
    
    
