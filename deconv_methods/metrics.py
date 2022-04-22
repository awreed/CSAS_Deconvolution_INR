import torch
import numpy as np
from deconv_methods.models import MLP_FF2D_MLP, FourierFeaturesVector
import collections
from sim_csas_package.utils import L1_reg, L2_reg, grad_reg, TV, save_img, normalize, imwrite,drc, save_sas_plot, c2g, g2c
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import lpips

class INRRecon:
  def __init__(self, RP):
    self.RP = RP

  def norm(self, x):
    return 2*((x - x.min())/(x.max() - x.min())) - 1

  def energy(self, x):
    return torch.sum(x.abs()**2)


  def recon(self, kappa, nf, max_iter, save_every, sim, psf, crop_size, lr, gt_img, img, save_name):
    assert img.dtype == np.complex128
    assert psf.dtype == np.complex128

    assert isinstance(sim, bool)

    SAVE_EVERY = save_every

    GRID_SIZE = self.RP.pix_dim_bf[0]

    # Reshape the PSF and target variables
    # [1, 1, PSF_H, PSF_W]
    psf = torch.from_numpy(psf).to(self.RP.dev)[None, None, ...]
    # [Num bf pixels]
    y = torch.from_numpy(img).to(self.RP.dev)
    y = y / torch.sqrt(self.energy(y))

    # Define the network
    self.RP.nf = nf
    model = MLP_FF2D_MLP(RP=self.RP, out_ch=2,
        act='none').double().to(self.RP.dev)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

    # Define the INR input coorrdinates
    x = np.linspace(-1, 1, self.RP.pix_dim_bf[0], endpoint=True)
    scene_xy = np.stack(np.meshgrid(x, x), axis=-1)
    scene_xy = scene_xy[scene_xy[..., 0]**2 + scene_xy[..., 1]**2 <= 1]
    scene_xy = torch.from_numpy(scene_xy).to(self.RP.dev)

    # Create module to compute Fourier Feature of input coordinates
    gr_ff = FourierFeaturesVector(2, self.RP.nf, kappa).to(self.RP.dev).double()
    loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)

    # Compute the Fourier Features
    scene_ff = gr_ff(scene_xy)

    psnrs = []
    ssims = []
    lpips = []
    images = []

    for epoch in range(0, max_iter):
      # Clear torch optimizer
      optimizer.zero_grad()

      # get the complex (2 channel) network output
      x = model(scene_ff).squeeze()
      x_real = x[..., 0].squeeze()
      x_imag = x[..., 1].squeeze()

      x_complex = torch.complex(real=x_real, imag=x_imag)

      # the network outputs a circular grid of points, we snap them to this square grid to play with well with the
      # the convolution function.
      x_complex_grid = torch.zeros((GRID_SIZE, GRID_SIZE),
          dtype=torch.complex128).view(-1).to(x_complex.device)
      x_complex_grid[self.RP.circle_indeces] = x_complex
      x_complex_grid = x_complex_grid.reshape((GRID_SIZE, GRID_SIZE))[None,
          None, ...]

      # Perform the convolution.
      x_conv_real = torch.nn.functional.conv2d(x_complex_grid.real, psf.real,
          padding='same').squeeze()
      x_conv_imag = torch.nn.functional.conv2d(x_complex_grid.imag, psf.real,
          padding='same').squeeze()

      # Normalize the predicted result before computing the loss
      pred = torch.complex(real=x_conv_real, imag=x_conv_imag)
      pred = pred.view(-1)[self.RP.circle_indeces]
      pred = pred / torch.sqrt(self.energy(pred))

      # Compute the loss on the real and imaginary parts. 'Sum' reduction tends to help the gradient descent converge on
      # both real and imaginary parts of data.
      loss = torch.nn.functional.mse_loss(pred.real,
        y.real, reduction='sum') + torch.nn.functional.mse_loss(pred.imag,
            y.imag, reduction='sum')

      # Backpropagate
      loss.backward()
      optimizer.step()

      if epoch % SAVE_EVERY == 0 or epoch == max_iter - 1:
        x = torch.sqrt(x_real**2 + x_imag**2)

        print("Epoch:", epoch, "\t", "Loss:", loss.item())

        deconv_scene = x.squeeze().detach().cpu().numpy()
        deconv_scene = c2g(deconv_scene, self.RP.circle_indeces, GRID_SIZE)

        if crop_size:
          deconv_scene = deconv_scene[crop_size//2:-crop_size//2,
            crop_size//2:-crop_size//2]

          deconv_scene, indeces = g2c(deconv_scene)

        if sim:
          images.append(normalize(deconv_scene))

          psnr_est = peak_signal_noise_ratio(normalize(gt_img),
              normalize(deconv_scene))
          ssim_est = structural_similarity(normalize(gt_img),
              normalize(deconv_scene))
          gt = torch.from_numpy(self.norm(gt_img.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)
          est = torch.from_numpy(self.norm(deconv_scene.squeeze()))[None, None,
              ...].repeat(1, 3, 1, 1).to(self.RP.dev)
          percep = loss_fn_alex(gt, est).item()

          psnrs.append(psnr_est)
          ssims.append(ssim_est)
          lpips.append(percep)

          print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep:", percep)

          save_sas_plot(deconv_scene, os.path.join(save_name, "deconv_img_" + str(epoch) + '.png'))
          np.save(os.path.join(save_name, "deconv_img_" + str(epoch) + '.npy'), deconv_scene)
          np.save(os.path.join(save_name, "complex_DAS_pred_" + str(epoch) + '.npy'), pred.detach().cpu().numpy())

        else:
          save_sas_plot(deconv_scene, os.path.join(save_name, "deconv_img_" + str(epoch) + '.png'))
          np.save(os.path.join(save_name, "deconv_img_" + str(epoch) + '.npy'), deconv_scene)
          np.save(os.path.join(save_name, "complex_DAS_pred_" + str(epoch) + '.npy'), pred.detach().cpu().numpy())

    return images, psnrs, ssims, lpips