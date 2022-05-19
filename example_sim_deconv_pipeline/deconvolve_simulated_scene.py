import sys
import numpy as np
sys.path.append('../')
from sim_csas_package import SimulateMeasurements, CreatePSF
from deconv_methods.deconv_methods import DeconvMethods
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
from sim_csas_package.utils import natural_keys, normalize

if __name__ == '__main__':
    sim_config = 'config/simulation.ini'
    sys_config = 'config/system_parameters.ini'
    deconv_config = 'config/deconv.ini'
    save_img_dir = 'sim_images'
    save_data_dir = 'sim_data'
    deconv_dir = 'deconv_dir'

    print("Simulating Scenes")
    # Simulate measurements using the specifications given in the sim_config file
    SM = SimulateMeasurements(sim_config=sim_config, sys_config=sys_config,
                              save_img_dir=save_img_dir, save_data_dir=save_data_dir)
    SM.run()

    print("Creating PSF")
    PSF = CreatePSF(sys_config=sys_config, save_img_dir=save_img_dir, save_data_dir=save_data_dir)
    PSF.run()

    to_be_deconvolved = [{'scene': np.load('sim_data/beamformed_scatterers_0.npy'),
                          'psf': np.load('sim_data/psf.npy'),
                          'gt': np.load('sim_data/gt0.npy')},

                         {'scene': np.load('sim_data/beamformed_scatterers_1.npy'),
                          'psf': np.load('sim_data/psf.npy'),
                          'gt': np.load('sim_data/gt1.npy')}]

    # Pass in the scene and the psf for the scene.
    DM = DeconvMethods(deconv_config=deconv_config,
                       to_be_deconvolved=to_be_deconvolved,
                       deconv_dir=deconv_dir,
                       circular=True,
                       device='cuda:0')

    DM.run_all_methods()

    ### PLOTTING #########################
    plt.rcParams['axes.grid'] = False
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 5}
    matplotlib.rc('font', **font)

    deconv_image_dirs = ['image0', 'image1']

    # For all image directories
    for i, image in enumerate(deconv_image_dirs):
        fig = plt.figure(figsize=(7, 1.0))
        big_gs = fig.add_gridspec(1, len(DM.use_methods) + 3, wspace=0.01, hspace=0.08, left=0.00, right=1.0,
                                  width_ratios=[1 for x in range(len(DM.use_methods) + 2)] + [.07])

        # Load the ground truth and DAS images
        gt_img = np.load(os.path.join(save_data_dir, 'gt' + str(i) + '.npy'))
        bf_img = np.load(os.path.join(save_data_dir, 'beamformed_scatterers_' + str(i) + '.npy'))
        bf_img = np.abs(bf_img)

        gt_img = normalize(gt_img)
        bf_img = normalize(bf_img)

        # plot them
        gt_ax = fig.add_subplot(big_gs[0, 0])
        y = gt_ax.imshow(gt_img, aspect='auto')
        gt_ax.axis('off')
        gt_ax.set_title('Ground Truth')

        bf_ax = fig.add_subplot(big_gs[0, 1])
        bf_ax.imshow(bf_img, aspect='auto')
        bf_ax.axis('off')
        bf_ax.set_title('DAS Reconstruction')

        # colorbar
        cb_ax = fig.add_subplot(big_gs[:, -1])
        fig.colorbar(y, cax=cb_ax, aspect=10, label='Linear Magnitude')

        # Create bar plots
        fig_psnr = plt.figure()
        ax_psnr = fig_psnr.add_axes([0, 0, 1, 1])

        fig_ssim = plt.figure()
        ax_ssim = fig_ssim.add_axes([0, 0, 1, 1])

        fig_lpips = plt.figure()
        ax_lpips = fig_lpips.add_axes([0, 0, 1, 1])

        names = []
        max_psnr = []
        max_ssim = []
        min_lpips = []

        # Plot the result of each deconvolution method...
        for j, meth in enumerate(DM.use_methods):
            # Get the PSNR, SSIM, and LPIPS scores for the method
            psnr_list = np.load(os.path.join(deconv_dir, image, meth['name'], 'psnrs.npy'))
            ssim_list = np.load(os.path.join(deconv_dir, image, meth['name'], 'ssims.npy'))
            lpips_list = np.load(os.path.join(deconv_dir, image, meth['name'], 'lpips.npy'))

            # List of all deconvolved images
            tmp_dir = os.path.join(deconv_dir, image, meth['name'])
            deconvolved_images = glob.glob(tmp_dir + '/' + 'deconv*.npy')
            deconvolved_images.sort(key=natural_keys)

            # Get index of max PSNR value
            ind = np.argmax(psnr_list)

            # Get all scores and image at max PSNR index
            psnr = psnr_list[ind]
            ssim = ssim_list[ind]
            lpips = lpips_list[ind]

            deconv_img = np.load(deconvolved_images[ind])

            deconv_img = normalize(deconv_img)

            ax = fig.add_subplot(big_gs[0, 2 + j])
            ax.imshow(deconv_img, aspect='auto')
            ax.axis('off')
            ax.set_title(meth['name'])

            max_psnr.append(psnr)
            max_ssim.append(ssim)
            min_lpips.append(lpips)
            names.append(meth['name'])

        ax_psnr.bar(names, max_psnr)
        ax_psnr.set_xlabel('Method')
        ax_psnr.set_ylabel('PSNR')
        ax_psnr.set_title('PSNR scores (Higher is better)')

        ax_ssim.bar(names, max_ssim)
        ax_ssim.set_xlabel('Method')
        ax_ssim.set_ylabel('SSIM')
        ax_ssim.set_title('SSIM scores (Higher is better)')

        ax_lpips.bar(names, min_lpips)
        ax_lpips.set_xlabel('Method')
        ax_lpips.set_ylabel('LPIPs')
        ax_lpips.set_title('LPIPS scores (Lower is better)')

        print("Saving image results to", os.path.join(deconv_dir, image), "directory")

        fig_psnr.savefig(os.path.join(deconv_dir, image, 'psnr_bar_plot.png'), dpi=300, bbox_inches='tight')
        fig_ssim.savefig(os.path.join(deconv_dir, image, 'ssim_bar_plot.png'), dpi=300, bbox_inches='tight')
        fig_lpips.savefig(os.path.join(deconv_dir, image, 'lpips_bar_plot.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(deconv_dir, image, 'deconvolution_results.png'), dpi=300, bbox_inches='tight')





