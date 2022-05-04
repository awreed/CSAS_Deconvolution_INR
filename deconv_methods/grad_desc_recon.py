import torch
import numpy as np
from sim_csas_package.utils import grad_reg, TV, normalize, save_sas_plot, c2g, g2c
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import lpips


class GradDescRecon:
    def __init__(self, device, circular):
        self.device = device
        self.circular = circular

    def norm(self, x):
        return 2 * ((x - x.min()) / (x.max() - x.min())) - 1

    def energy(self, x):
        return torch.sum(x.abs() ** 2)

    def recon(self, lr, momentum, reg, reg_weight, max_iter, save_every, img, psf, gt_img, save_name):
        assert img.dtype == np.complex128
        assert psf.dtype == np.complex128

        assert reg in ['none', 'l1', 'l2', 'grad_reg', 'tv']

        reg_fn = None

        if reg == 'none':
            reg_fn = lambda x: torch.tensor([0]).to(self.device).double()
        elif reg == 'grad_reg':
            reg_fn = grad_reg
        elif reg == 'tv':
            reg_fn = TV

        x_shape, y_shape = img.shape[0], img.shape[1]

        if self.circular:
            assert img.shape[0] == img.shape[1], "Image must be square (H == W) to do circular reconstruction."
            _, ind = g2c(img)
            img = img.ravel()[ind]
            self.ind = ind
        else:
            img = img.ravel()

        psf = torch.from_numpy(psf).to(self.device)[None, None, ...]
        psf = psf / torch.sqrt(self.energy(psf))

        target = torch.from_numpy(img).to(self.device)
        target = target / torch.sqrt(self.energy(target))

        x = np.linspace(0, 1., x_shape, endpoint=True)
        y = np.linspace(0, 1., y_shape, endpoint=True)
        scene_xy = np.stack(np.meshgrid(x, y), axis=-1)

        scene_xy = torch.from_numpy(scene_xy).to(self.device).double()

        loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.device)

        if self.circular:
            mask = torch.zeros((x_shape, y_shape),
                               dtype=torch.complex128).to(self.device)[None, None, ...]
            mask.view(-1)[self.ind] = 1

        x_real = scene_xy[..., 0].squeeze()
        x_imag = scene_xy[..., 1].squeeze()

        x_complex = torch.complex(real=x_real, imag=x_imag)[None, None,
                                                            ...].requires_grad_()

        optimizer = torch.optim.SGD([x_complex], lr=lr, momentum=momentum)

        psnrs = []
        ssims = []
        perceps = []
        images = []

        for epoch in range(0, max_iter):
            optimizer.zero_grad()

            if self.circular:
                _x_complex = x_complex * mask
            else:
                _x_complex = x_complex

            x_conv_real = torch.nn.functional.conv2d(_x_complex.real, psf.real,
                                                     padding='same').squeeze()
            x_conv_imag = torch.nn.functional.conv2d(_x_complex.imag, psf.real,
                                                     padding='same').squeeze()

            pred = torch.complex(real=x_conv_real, imag=x_conv_imag)

            if self.circular:
                _pred = pred.view(-1)[self.ind]
                __pred = _pred / torch.sqrt(self.energy(_pred))
            else:
                _pred = pred.view(-1)
                __pred = _pred / torch.sqrt(self.energy(_pred))

            reg_term = reg_weight * reg_fn(_x_complex.abs().squeeze())

            loss = torch.nn.functional.mse_loss(__pred.real, target.real,
                                                reduction='sum') \
                   + torch.nn.functional.mse_loss(__pred.imag, target.imag,
                                                  reduction='sum') \
                   + reg_term

            loss.backward()
            optimizer.step()

            if epoch % save_every == 0 or epoch == max_iter - 1:
                print("Epoch:", epoch, "\t", "Loss:", loss.item())

                deconv_scene = _x_complex.abs().squeeze().detach().cpu().numpy()

                deconv_scene = deconv_scene.reshape((x_shape, y_shape))

                images.append(normalize(deconv_scene))

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

                save_sas_plot(deconv_scene, os.path.join(save_name, "deconv_img_" + str(epoch) + '.png'))
                np.save(os.path.join(save_name, "deconv_img_" + str(epoch) + '.npy'), deconv_scene)
                np.save(os.path.join(save_name, "complex_DAS_pred_" + str(epoch) + '.npy'), pred.detach().cpu().numpy())

        return images, psnrs, ssims, perceps
