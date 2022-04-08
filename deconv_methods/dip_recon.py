import torch
import numpy as np
from models import MLP_FF2D_MLP, FourierFeaturesVector
import collections
from utils import L1_reg, L2_reg, grad_reg, TV, save_img, normalize, imwrite, save_sas_plot
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import skimage.transform
import cv2
import lpips
from dip_models import *
from dip_utils import *
import pdb
from dip_utils.common_utils import *


class DIP_Recon:
  def __init__(self, RP):
    self.RP = RP

  def norm(self, x):
    return 2*((x - x.min())/(x.max() - x.min())) - 1

  def energy(self, x):
    return torch.sum(x.abs()**2)


  def recon(self, max_len, max_iter, reg, reg_weight, save_every, sim, psf, crop_size, gt_img, img,
      save_name):

    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    assert reg in ['none', 'l1', 'l2', 'grad_reg', 'tv']

    reg_fn = None

    if reg == 'none':
      reg_fn = lambda x: torch.tensor([0]).to(self.RP.dev).double()
    elif reg == 'l1':
      reg_fn = L1_reg
    elif reg == 'l2':
      reg_fn = L2_reg
    elif reg == 'grad_reg':
      reg_fn = grad_reg
    elif reg == 'tv':
      reg_fn = TV

    SAVE_EVERY = save_every

    GRID_SIZE = self.RP.pix_dim_bf[0]

    #img = torch.from_numpy(img).to(self.RP.dev)
    #scale = torch.sqrt(1 / torch.max(img.abs()))
    #img = scale * img
    #img = img.detach().cpu().numpy()

    psf = torch.from_numpy(psf).to(self.RP.dev)[None, None, ...]
    y = torch.from_numpy(img).to(self.RP.dev)

    #psf = psf / torch.sqrt(self.energy(psf))
    y = y / torch.sqrt(self.energy(y))

    
    MAX_LEN = max_len
    d = collections.deque(maxlen=MAX_LEN)
    ssim_opt = collections.deque(maxlen=MAX_LEN)
    psnr_opt = collections.deque(maxlen=MAX_LEN)
    percep_opt = collections.deque(maxlen=MAX_LEN)
    images = collections.deque(maxlen=MAX_LEN)

    pw = 1

    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)


    #### Initialize DIP model ####
    weight = 0.0
    input_depth = 32
    pad = 'zeros'
    INPUT = 'noise'
    exp_weight = 0.99
    OPTIMIZER = 'adam'
    # real data 0.5e1
    LR = 1e-2
    OPT_OVER = 'net'

    model = get_net(input_depth, 'skip', pad, n_channels=2,
        act_fun='Swish',
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode='bilinear', need_sigmoid=False, need_bias=True).double().to(self.RP.dev)

    noise_input = get_noise(input_depth, INPUT, (GRID_SIZE,
      GRID_SIZE)).double().to(self.RP.dev).detach()

    optimizer = torch.optim.Adam(get_params(OPT_OVER, model, noise_input), lr=LR)

    mask = torch.zeros((GRID_SIZE, GRID_SIZE),
        dtype=torch.complex128).to(self.RP.dev) 
    mask.view(-1)[self.RP.circle_indeces] = 1

    for epoch in range(1, 100000000):
      optimizer.zero_grad()

      x = model(noise_input).squeeze()
      x_real = x[0, ...].squeeze()
      x_imag = x[1, ...].squeeze()

      x_complex = torch.complex(real=x_real, imag=x_imag)[None, None, ...]

      x_complex_circle = mask*x_complex

      x_conv_real = torch.nn.functional.conv2d(x_complex_circle.real, psf.real,
          padding='same').squeeze()
      x_conv_imag = torch.nn.functional.conv2d(x_complex_circle.imag, psf.real,
          padding='same').squeeze()


      pred = torch.complex(real=x_conv_real, imag=x_conv_imag)
      pred = pred.view(-1)[self.RP.circle_indeces]
      pred = pred / torch.sqrt(self.energy(pred))

      pixel_loss = torch.nn.functional.mse_loss(pred.real, y.real,
        reduction='sum') + torch.nn.functional.mse_loss(pred.imag, y.imag, 
            reduction='sum')

      loss = pixel_loss

      loss.backward()
      optimizer.step()

      if epoch % SAVE_EVERY == 0:

        x = torch.sqrt(x_complex_circle.real**2 + x_complex_circle.imag**2)

        deconv_scene = x.squeeze().detach().cpu().numpy()

        print("Epoch:", epoch, "\t", "Loss:", pixel_loss.item())

        
        if crop_size:
          deconv_scene = deconv_scene[crop_size//2:-crop_size//2,
            crop_size//2:-crop_size//2]

        if sim:

          save_sas_plot(deconv_scene, os.path.join('tmp', 'deconv_scene'
            + str(epoch) + '.png'))

          images.append(normalize(deconv_scene))

          #imwrite(deconv_scene, os.path.join('plots_200/deconv_scene' + str(epoch)
          #  + '.png'))
          pred = np.absolute(pred.squeeze().detach().cpu().numpy())
          #imwrite(pred, os.path.join('plots/pred' + str(epoch) + '.png'))

          psnr_est = peak_signal_noise_ratio(normalize(gt_img),
              normalize(deconv_scene))

          ssim_est = structural_similarity(normalize(gt_img),
              normalize(deconv_scene))
         
          gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)
          est = torch.from_numpy(self.norm(deconv_scene.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)

          percep = loss_fn_alex(gt, est).item()

          print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep:", percep)

          d.append(psnr_est)
          ssim_opt.append(ssim_est)
          psnr_opt.append(psnr_est)
          percep_opt.append(percep)

          if len(d) == MAX_LEN:
            if d[-1] < d[0] or epoch > 2000:
              val = max(psnr_opt)
              max_psnr = psnr_opt[psnr_opt.index(val)]
              max_ssim = ssim_opt[psnr_opt.index(val)]
              max_percep = percep_opt[psnr_opt.index(val)]
              deconv_scene = images[psnr_opt.index(val)]
           
              #print("Max PSNR", max_psnr)
              #print("Max SSIM", max_ssim)
              #print("Max Percep", max_percep)
              #imwrite(deconv_scene, os.path.join(SAVE_IMG_DIR,
              #'deconv_inr' + str(count) + '.png'))
              return max_psnr, max_ssim, max_percep, deconv_scene

        else:
          if epoch > MAX_LEN:
            return deconv_scene
          else:
            save_sas_plot(deconv_scene, os.path.join(save_name, str(epoch) + '.png'))
            np.save(os.path.join(save_name, str(epoch) + '.npy'), deconv_scene)

            abs_pred = torch.sqrt(x_conv_real**2
                + x_conv_imag**2).squeeze().detach().cpu().numpy()
            save_sas_plot(abs_pred, os.path.join(save_name, 'pred' + str(epoch)
              + '.png'))
            np.save(os.path.join(save_name, 'pred' + str(epoch) + '.npy'),
                abs_pred)


        



  

