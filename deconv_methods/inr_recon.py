import torch
import numpy as np
from deconv_methods.models import MLP_FF2D_MLP, FourierFeaturesVector
from sim_csas_package.utils import normalize, save_sas_plot, c2g, g2c
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class INRRecon:
    def __init__(self, device, circular=False):
        self.device = device
        self.circular = circular
        self.ind = None

    def norm(self, x):
        return 2 * ((x - x.min()) / (x.max() - x.min())) - 1

    def energy(self, x):
        return torch.sum(x.abs() ** 2)

    def recon(self, kappa, nf, lr, max_iter, save_every, img, psf, gt_img, save_name):
        assert img.dtype == np.complex128, "Cast DAS input and PSF input to np.complex128 type"
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
            img = img.ravel()[ind]
            self.ind = ind
        else:
            img = img.ravel()

        # Reshape the PSF and target variables
        # [1, 1, PSF_H, PSF_W]
        psf = torch.from_numpy(psf).to(self.device)[None, None, ...]
        psf = psf / torch.sqrt(self.energy(psf))
        # [Num bf pixels]
        target = torch.from_numpy(img).to(self.device)
        target = target / torch.sqrt(self.energy(target))

        # Define the network
        model = MLP_FF2D_MLP(nf=nf, out_ch=2,
                             act='none').double().to(self.device)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

        # Define the INR input coorrdinates
        x = np.linspace(-1, 1, x_shape, endpoint=True)
        y = np.linspace(-1, 1, y_shape, endpoint=True)
        scene_xy = np.stack(np.meshgrid(x, y), axis=-1)

        if self.circular:
            scene_xy = scene_xy[scene_xy[..., 0] ** 2 + scene_xy[..., 1] ** 2 <= 1]
        else:
            scene_xy = scene_xy.reshape(-1, 2)

        scene_xy = torch.from_numpy(scene_xy).to(self.device)

        # Create module to compute Fourier Feature of input coordinates
        gr_ff = FourierFeaturesVector(2, nf, kappa).to(self.device).double()
        loss_fn_alex = lpips.LPIPS(net='alex').double().to(self.device)

        # Compute the Fourier Features
        scene_ff = gr_ff(scene_xy)

        psnrs = []
        ssims = []
        perceps = []
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
            x_complex_grid = torch.zeros((x_shape, y_shape),
                                             dtype=torch.complex128).view(-1).to(x_complex.device)
            if self.circular:
                x_complex_grid[self.ind] = x_complex
            else:
                x_complex_grid = x_complex

            x_complex_grid = x_complex_grid.reshape((x_shape, y_shape))[None, None, ...]

            # changed the convolution. Changed the

            # Perform the convolution.
            x_conv_real = torch.nn.functional.conv2d(x_complex_grid.real, psf.real,
                                                     padding='same').squeeze()
            x_conv_imag = torch.nn.functional.conv2d(x_complex_grid.imag, psf.imag,
                                                     padding='same').squeeze()

            # Normalize the predicted result before computing the loss
            pred = torch.complex(real=x_conv_real, imag=x_conv_imag)

            if self.circular:
                pred = pred.view(-1)[self.ind]
            else:
                pred = pred.view(-1)

            pred = pred / torch.sqrt(self.energy(pred))

            # Compute the loss on the real and imaginary parts. 'Sum' reduction tends to help the gradient descent
            # converge on both real and imaginary parts of data.
            loss = torch.nn.functional.mse_loss(pred.real,
                                                target.real, reduction='sum') + torch.nn.functional.mse_loss(pred.imag,
                                                                                                        target.imag,
                                                                                                        reduction='sum')

            # Backpropagate
            loss.backward()
            optimizer.step()

            if epoch % save_every == 0 or epoch == max_iter - 1:
                x = torch.sqrt(x_real ** 2 + x_imag ** 2)

                print("Epoch:", epoch, "\t", "Loss:", loss.item())

                deconv_scene = x.squeeze().detach().cpu().numpy()
                pred = pred.squeeze().detach().cpu().numpy()

                if self.circular:
                    deconv_scene = c2g(deconv_scene, self.ind, x_shape, y_shape)
                    pred = c2g(pred, self.ind, x_shape, y_shape)
                else:
                    deconv_scene = deconv_scene.reshape((x_shape, y_shape))
                    pred = pred.reshape((x_shape, y_shape))

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
                save_sas_plot(np.abs(pred), os.path.join(save_name, "complex_DAS_pred_" +
                                                                                str(epoch) + '.png'))
                np.save(os.path.join(save_name, "deconv_img_" + str(epoch) + '.npy'), deconv_scene)
                np.save(os.path.join(save_name, "complex_DAS_pred_" + str(epoch) + '.npy'), pred)

        return images, psnrs, ssims, perceps
