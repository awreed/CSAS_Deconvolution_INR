import torch
import numpy as np
from sim_csas_package.utils import grad_reg, TV, normalize, save_sas_plot, c2g
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import lpips


class GradDescRecon:
    def __init__(self, RP):
        self.RP = RP

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
            reg_fn = lambda x: torch.tensor([0]).to(self.RP.dev).double()
        elif reg == 'grad_reg':
            reg_fn = grad_reg
        elif reg == 'tv':
            reg_fn = TV

        GRID_SIZE = self.RP.pix_dim_bf[0]

        psf = torch.from_numpy(psf).to(self.RP.dev)[None, None, ...]
        y = torch.from_numpy(img).to(self.RP.dev)
        y = y / torch.sqrt(self.energy(y))

        x = np.linspace(0, .1, GRID_SIZE, endpoint=True)
        x = np.stack(np.meshgrid(x, x), axis=-1)

        x = torch.from_numpy(x).to(self.RP.dev).double()

        loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.RP.dev)

        mask = torch.zeros((GRID_SIZE, GRID_SIZE),
                           dtype=torch.complex128).to(self.RP.dev)[None, None, ...]
        mask.view(-1)[self.RP.circle_indeces] = 1

        x_real = x[..., 0].squeeze()
        x_imag = x[..., 1].squeeze()
        print(x_real.shape)
        print(x_real[GRID_SIZE // 2, GRID_SIZE // 2])
        x_complex = torch.complex(real=x_real, imag=x_imag)[None, None,
                                                            ...].requires_grad_()

        optimizer = torch.optim.SGD([x_complex], lr=lr, momentum=momentum)

        psnrs = []
        ssims = []
        perceps = []
        images = []

        for epoch in range(0, max_iter):
            optimizer.zero_grad()

            x_complex_circle = x_complex * mask

            x_conv_real = torch.nn.functional.conv2d(x_complex_circle.real, psf.real,
                                                     padding='same').squeeze()
            x_conv_imag = torch.nn.functional.conv2d(x_complex_circle.imag, psf.real,
                                                     padding='same').squeeze()

            pred = torch.complex(real=x_conv_real, imag=x_conv_imag)
            pred_sub = pred.view(-1)[self.RP.circle_indeces]
            pred_norm = pred_sub / torch.sqrt(self.energy(pred_sub))

            reg_term = reg_weight * reg_fn(x_complex_circle.abs().squeeze())

            loss = torch.nn.functional.mse_loss(pred_norm.real, y.real,
                                                reduction='sum') \
                   + torch.nn.functional.mse_loss(pred_norm.imag, y.imag,
                                                  reduction='sum') \
                   + reg_term

            loss.backward()
            optimizer.step()

            if epoch % save_every == 0 or epoch == max_iter - 1:
                x = torch.sqrt(x_real ** 2 + x_imag ** 2)

                print("Epoch:", epoch, "\t", "Loss:", loss.item())

                deconv_scene = x.squeeze().detach().cpu().numpy()
                deconv_scene = c2g(deconv_scene, self.RP.circle_indeces, GRID_SIZE)

                images.append(normalize(deconv_scene))

                if gt_img is not None:
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
                    perceps.append(percep)

                    print("PSNR EST:", psnr_est, "SSIM:", ssim_est, "Percep:", percep)

                save_sas_plot(deconv_scene, os.path.join(save_name, "deconv_img_" + str(epoch) + '.png'))
                np.save(os.path.join(save_name, "deconv_img_" + str(epoch) + '.npy'), deconv_scene)
                np.save(os.path.join(save_name, "complex_DAS_pred_" + str(epoch) + '.npy'), pred.detach().cpu().numpy())

        return images, psnrs, ssims, perceps
