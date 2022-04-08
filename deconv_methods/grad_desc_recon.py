import torch
import numpy as np
from models import MLP_FF2D_MLP, FourierFeaturesVector
import collections
from utils import L1_reg, L2_reg, grad_reg, TV, save_img, normalize, imwrite, save_sas_plot, c2g, g2c
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import lpips


class GradDescRecon:
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

    if crop_size:
      gt_img = gt_img[crop_size//2:-crop_size//2, crop_size//2:-crop_size//2]

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

    #x = np.random.uniform(low=0, high=.1, size=(GRID_SIZE, GRID_SIZE, 2))
    x = np.linspace(0, .1, GRID_SIZE, endpoint=True)
    x = np.stack(np.meshgrid(x, x), axis=-1)
    
    # = np.ones((GRID_SIZE, GRID_SIZE, 2))

    x = torch.from_numpy(x).to(self.RP.dev).double()

    MAX_LEN = max_len
    d = collections.deque(maxlen=MAX_LEN)
    ssim_opt = collections.deque(maxlen=MAX_LEN)
    psnr_opt = collections.deque(maxlen=MAX_LEN)
    percep_opt = collections.deque(maxlen=MAX_LEN)
    images = collections.deque(maxlen=MAX_LEN)

    # was at 10 for initial results
    
    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)

    mask = torch.zeros((GRID_SIZE, GRID_SIZE),
        dtype=torch.complex128).to(self.RP.dev)[None, None, ...]
    mask.view(-1)[self.RP.circle_indeces] = 1

    x_real = x[..., 0].squeeze()
    x_imag = x[..., 1].squeeze()
    print(x_real.shape)
    print(x_real[GRID_SIZE//2, GRID_SIZE//2])
    x_complex = torch.complex(real=x_real, imag=x_imag)[None, None,
        ...].requires_grad_()

    optimizer = torch.optim.SGD([x_complex], lr=10, momentum=0.9)

    #print("Starting optimization")
    SAVE_EVERY = save_every
    for epoch in range(0, 100000000):
      optimizer.zero_grad()
      #x_complex = x_complex.detach()
      #x_complex.requires_grad = True

      #print(x_complex.real[0, 0, GRID_SIZE//2, GRID_SIZE//2])

      x_complex_circle = x_complex*mask

      #x_norm = torch.nn.functional.relu(x)
      x_conv_real = torch.nn.functional.conv2d(x_complex_circle.real, psf.real,
          padding='same').squeeze()
      x_conv_imag = torch.nn.functional.conv2d(x_complex_circle.imag, psf.real,
          padding='same').squeeze()

      pred = torch.complex(real=x_conv_real, imag=x_conv_imag)
      pred_sub = pred.view(-1)[self.RP.circle_indeces]
      pred_norm = pred_sub / torch.sqrt(self.energy(pred_sub))

      reg_term = reg_weight * reg_fn(x_complex_circle.abs().squeeze())

      loss = torch.nn.functional.mse_loss(pred_norm.real, y.real,
          reduction='sum')\
        + torch.nn.functional.mse_loss(pred_norm.imag, y.imag,
            reduction='sum')\
        + reg_term

      #print(loss.item())

      loss.backward()
      optimizer.step()

      if epoch % SAVE_EVERY == 0:


        print("Epoch:", epoch, "\t", "Loss:", loss.item(), "Reg",
            reg_term.item())

        deconv_scene = x_complex_circle.abs().squeeze().detach().cpu().numpy()
        
        if crop_size:
          deconv_scene = deconv_scene[crop_size//2:-crop_size//2,
            crop_size//2:-crop_size//2]

          deconv_scene, indeces = g2c(deconv_scene)

        #imwrite(deconv_scene, 'plots_200/deconv_gd' + str(epoch) + '.png')

        if sim:
          images.append(normalize(deconv_scene))

          save_sas_plot(normalize(deconv_scene), os.path.join('tmp',
            'deconv_scene' + str(epoch) + '.png'))

          psnr_est = peak_signal_noise_ratio(normalize(gt_img),
              normalize(deconv_scene))

          ssim_est = structural_similarity(normalize(gt_img),
              normalize(deconv_scene))


          gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)
          est = torch.from_numpy(self.norm(deconv_scene.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)

          percep = loss_fn_alex(gt, est).item()

          print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep", percep)

          d.append(psnr_est)
          ssim_opt.append(ssim_est)
          psnr_opt.append(psnr_est)
          percep_opt.append(percep)

          if len(d) == MAX_LEN:
            if d[-1] < d[0] or epoch > max_iter:
              val = max(psnr_opt)
              max_psnr = psnr_opt[psnr_opt.index(val)]
              max_ssim = ssim_opt[psnr_opt.index(val)]
              max_percep = percep_opt[psnr_opt.index(val)]
              deconv_scene = images[psnr_opt.index(val)]
              #print("Max PSNR", max_psnr)
              #print("Max SSIM", max_ssim)
              #print("MAX percep", max_percep)
              #imwrite(deconv_scene, os.path.join(SAVE_IMG_DIR,
              #'deconv_gd' + str(count) + '.png'))
              return max_psnr, max_ssim, max_percep, deconv_scene
        else:
          if epoch > MAX_LEN:
            return deconv_scene
          else:
            save_sas_plot(deconv_scene, os.path.join(save_name, str(epoch)
              + '.png'))
            abs_val = torch.sqrt(x_conv_real**2
                + x_conv_imag**2).squeeze().detach().cpu().numpy()
            save_sas_plot(abs_val, os.path.join(save_name, 'pred' + str(epoch)
              + '.png'))
            
            #save_sas_plot(x_conv_real.detach().cpu().numpy(),
            #    os.path.join(save_name, 'real' + str(epoch) + '.png'))
            #save_sas_plot(x_conv_imag.detach().cpu().numpy(),
            #    os.path.join(save_name, 'imag' + str(epoch) + '.png'))
            np.save(os.path.join(save_name, str(epoch) + '.npy'), deconv_scene)
            np.save(os.path.join(save_name, 'pred' + str(epoch) + '.npy'),
                abs_val)
            np.save(os.path.join(save_name, 'pred_real' + str(epoch) + '.npy'),
                x_conv_real.detach().cpu().numpy())
            np.save(os.path.join(save_name, 'pred_imag' + str(epoch) + '.npy'),
                x_conv_imag.detach().cpu().numpy())




  

