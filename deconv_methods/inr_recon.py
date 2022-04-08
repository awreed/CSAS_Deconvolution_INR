import torch
import numpy as np
from models import MLP_FF2D_MLP, FourierFeaturesVector
import collections
from utils import L1_reg, L2_reg, grad_reg, TV, save_img, normalize, imwrite,drc, save_sas_plot, c2g, g2c
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import skimage.transform
import cv2
import lpips
import matplotlib.pyplot as plt

class INR_Recon:
  def __init__(self, RP):
    self.RP = RP

  def norm(self, x):
    return 2*((x - x.min())/(x.max() - x.min())) - 1

  def energy(self, x):
    return torch.sum(x.abs()**2)


  def recon(self, sigma, max_len, max_iter, reg, reg_weight,
      save_every, sim, psf, crop_size, lr, gt_img, img, save_name):
    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    assert reg in ['none', 'l1', 'l2', 'grad_reg', 'tv']

    assert isinstance(sim, bool)

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

    # [1, 1, PSF_H, PSF_W]
    psf = torch.from_numpy(psf).to(self.RP.dev)[None, None, ...]
    # [Num bf pixels]
    y = torch.from_numpy(img).to(self.RP.dev)

    #psf = psf / torch.sqrt(self.energy(psf))
    y = y / torch.sqrt(self.energy(y))

    self.RP.nf = 256
    model = MLP_FF2D_MLP(RP=self.RP, out_ch=2,
        act='none').double().to(self.RP.dev)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

    # INR coordinates
    x = np.linspace(-1, 1, self.RP.pix_dim_bf[0], endpoint=True)
    scene_xy = np.stack(np.meshgrid(x, x), axis=-1)
    scene_xy = scene_xy[scene_xy[..., 0]**2 + scene_xy[..., 1]**2 <= 1]
    scene_xy = torch.from_numpy(scene_xy).to(self.RP.dev)

    #scene_xy = torch.from_numpy(scene_xy).to(self.RP.dev).permute(2, 0, 1)[None, ...]

    print(scene_xy.shape)
    print(psf.shape)
    print(y.shape)

    # -> input should be l, c
    gr_ff = FourierFeaturesVector(2, self.RP.nf, sigma).to(self.RP.dev).double()

    MAX_LEN = max_len
    d = collections.deque(maxlen=MAX_LEN)
    ssim_opt = collections.deque(maxlen=MAX_LEN)
    psnr_opt = collections.deque(maxlen=MAX_LEN)
    percep_opt = collections.deque(maxlen=MAX_LEN)
    images = collections.deque(maxlen=MAX_LEN)

    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)

    scene_ff = gr_ff(scene_xy)
    #scene_ff = scene_ff.squeeze().view(self.RP.nf*2, -1).permute(1, 0)

    print(scene_ff.shape)

    for epoch in range(1, 100000000):
      optimizer.zero_grad()

      x = model(scene_ff).squeeze()
      x_real = x[..., 0].squeeze()
      x_imag = x[..., 1].squeeze()
      
      #x_real = x_real / torch.sqrt(self.energy(x_real))
      #x_imag = x_imag / torch.sqrt(self.energy(x_imag))
      x_complex = torch.complex(real=x_real, imag=x_imag)
      #x_complex = x_complex / torch.sqrt(self.energy(x_complex))

      x_complex_grid = torch.zeros((GRID_SIZE, GRID_SIZE),
          dtype=torch.complex128).view(-1).to(x_complex.device)

      x_complex_grid[self.RP.circle_indeces] = x_complex
      x_complex_grid = x_complex_grid.reshape((GRID_SIZE, GRID_SIZE))[None,
          None, ...]
      #TODO Fix the x_conv_imag call and fix the pixel_loss function to include
      #imag and fix abs of x in if statement
      x_conv_real = torch.nn.functional.conv2d(x_complex_grid.real, psf.real,
          padding='same').squeeze()
      x_conv_imag = torch.nn.functional.conv2d(x_complex_grid.imag, psf.real,
          padding='same').squeeze()
      #x_conv_imag = torch.zeros_like(x_conv_real).to(x_conv_real.device)

      pred = torch.complex(real=x_conv_real, imag=x_conv_imag)
      pred = pred.view(-1)[self.RP.circle_indeces]

      pred = pred / torch.sqrt(self.energy(pred))
      #reg_term = reg_weight * reg_fn(torch.sqrt(x**2))

      pixel_loss = torch.nn.functional.mse_loss(pred.real,
        y.real, reduction='sum') + torch.nn.functional.mse_loss(pred.imag,  
            y.imag, reduction='sum')

      loss = pixel_loss #+ reg_term

      loss.backward()
      optimizer.step()

      if epoch % SAVE_EVERY == 0:
        print(x.min().item(), x.max().item())

        x = torch.sqrt(x_real**2 + x_imag**2)
        #x = torch.sqrt(x_real**2)
        #x = x_real + x_imag

        print("Epoch:", epoch, "\t", "Loss:", pixel_loss.item())

        deconv_scene = x.squeeze().detach().cpu().numpy()
        deconv_scene = c2g(deconv_scene, self.RP.circle_indeces, GRID_SIZE)

        #save_sas_plot(deconv_scene, os.path.join(save_name, 'full_size.png'))
        
        if crop_size:
          deconv_scene = deconv_scene[crop_size//2:-crop_size//2,
            crop_size//2:-crop_size//2]

          deconv_scene, indeces = g2c(deconv_scene)

        if sim:
          images.append(normalize(deconv_scene))

          #save_sas_plot(deconv_scene, os.path.join(save_name, 'deconv_scene'
          #  + str(epoch)+ '.png'))

          psnr_est = peak_signal_noise_ratio(normalize(gt_img),
              normalize(deconv_scene))

          
          ssim_est = structural_similarity(normalize(gt_img),
              normalize(deconv_scene))
         
          gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)
          est = torch.from_numpy(self.norm(deconv_scene.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)

          percep = loss_fn_alex(gt, est).item()

          d.append(psnr_est)

          print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep:", percep)

          #save_sas_plot(pred.real.squeeze().detach().cpu().numpy(),
          #  os.path.join(save_name, 'real_pred' + str(epoch) + '.png'))

          #save_sas_plot(pred.imag.squeeze().detach().cpu().numpy(),
          #  os.path.join(save_name, 'imag_pred' + str(epoch) + '.png'))

          save_sas_plot(deconv_scene, os.path.join(save_name, 'deconv'
            + str(epoch) + '.png'))
          np.save(os.path.join(save_name, 'deconv' + str(epoch) + '.png'),
              deconv_scene)
          #save_sas_plot(x_complex_grid.real.squeeze().detach().cpu().numpy(),
          #  os.path.join(save_name, 'x_real' + str(epoch) + '.png'))

          #save_sas_plot(x_complex_grid.imag.squeeze().detach().cpu().numpy(),
          #  os.path.join(save_name, 'x_imag' + str(epoch) + '.png'))

          #save_sas_plot(x_conv_real.squeeze().detach().cpu().numpy(),
          #  os.path.join(save_name, 'x_real_pred' + str(epoch) + '.png'))

          #save_sas_plot(x_conv_imag.squeeze().detach().cpu().numpy(),
          #  os.path.join(save_name, 'x_imag_pred' + str(epoch) + '.png'))

          #abs_pred = torch.sqrt(x_conv_real**2
          #    + x_conv_imag**2).squeeze().detach().cpu().numpy()
          #save_sas_plot(abs_pred, os.path.join(save_name, 'x_abs_pred'
          #  + str(epoch) + '.png'))



          ssim_opt.append(ssim_est)
          psnr_opt.append(psnr_est)
          percep_opt.append(percep)

          if len(d) == MAX_LEN:
            if d[-1] < d[0] or epoch > 3000:
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

            #save_sas_plot(x_complex_grid.real.squeeze().detach().cpu().numpy(),
            #  os.path.join(save_name, 'x_real' + str(epoch) + '.png'))

            #save_sas_plot(x_complex_grid.imag.squeeze().detach().cpu().numpy(),
            #  os.path.join(save_name, 'x_imag' + str(epoch) + '.png'))

            #save_sas_plot(x_conv_real.squeeze().detach().cpu().numpy(),
            #  os.path.join(save_name, 'x_real_pred' + str(epoch) + '.png'))

            #save_sas_plot(x_conv_imag.squeeze().detach().cpu().numpy(),
            #  os.path.join(save_name, 'x_imag_pred' + str(epoch) + '.png'))

            abs_pred = torch.sqrt(x_conv_real**2
              + x_conv_imag**2).squeeze().detach().cpu().numpy()
            
            save_sas_plot(abs_pred, os.path.join(save_name, 'x_abs_pred'
              + str(epoch) + '.png'))
            np.save(os.path.join(save_name, 'pred' + str(epoch) + '.npy'),
                abs_pred)
            np.save(os.path.join(save_name, 'pred_real' + str(epoch) + '.npy'),
                x_conv_real.detach().cpu().numpy())

            np.save(os.path.join(save_name, 'pred_imag' + str(epoch) + '.npy'),
                x_conv_imag.detach().cpu().numpy())






            #save_sas_plot(x_real.squeeze().detach().cpu().numpy(),
            #  os.path.join(save_name, 'x_real' + str(epoch) + '.png'))

            #save_sas_plot(x_imag.squeeze().detach().cpu().numpy(),
            #  os.path.join(save_name, 'x_imag' + str(epoch) + '.png'))

            #phase = np.arctan2(pred.imag.squeeze().detach().cpu().numpy(),
            #  pred.real.squeeze().detach().cpu().numpy())

            #save_sas_plot(phase, os.path.join(save_name, 'phase_pred'
            #  + str(epoch) + '.png'))
