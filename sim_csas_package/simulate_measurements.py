import torch
import os
import constants.constants as C
from sim_csas_package.render_parameters import RenderParameters
from sim_csas_package.utils import save_sas_plot, normalize, load_img_and_preprocess, c2g
from sim_csas_package.beamformer import Beamformer
from sim_csas_package.waveform_processing import delay_waveforms
import configparser
import numpy as np
from sim_csas_package.utils import process_sys_config


class SimulateMeasurements:
    def __init__(self, sim_config, sys_config, save_img_dir, save_data_dir):
        self.sim_config = configparser.ConfigParser()
        cond = self.sim_config.read(sim_config)
        assert sim_config in cond, "Failed to read simulation config (is path correct)"

        self.sys_config = configparser.ConfigParser()
        cond1 = self.sys_config.read(sys_config)
        assert sys_config in cond1, "Failed to read system config file (is path correct?)"

        self.save_img_dir = save_img_dir
        self.save_data_dir = save_data_dir

        if torch.cuda.is_available():
            self.dev = 'cuda:0'
        else:
            self.dev = 'cpu'
            print("Did not find gpu so using", self.dev)

    def run(self):
        with torch.no_grad():
            LOAD_IMG_DIR = self.sim_config[C.IO][C.LOAD_IMG_DIR]

            image_names = self.sim_config[C.IO][C.LOAD_IMG_NAMES]
            image_names = image_names.split(',')
            image_names = [num.strip() for num in image_names]

            sys = process_sys_config(self.sys_config)

            RP = RenderParameters(device=self.dev, Fs=sys['fs'], c=sys['c'],
                                  f_start=sys['f_start'], f_stop=sys['f_stop'],
                                  t_start=sys['t_start'], t_stop=sys['t_stop'],
                                  win_ratio=sys['win_ratio'])

            # define transducer positions relative to the scene
            RP.define_transducer_pos(theta_start=sys['theta_start'], theta_stop=sys['theta_stop'],
                                     theta_step=sys['theta_step'], r=sys['radius'], z_TX=sys['Z_TX'],
                                     z_RX=sys['Z_RX'])

            pix_dim = sys['pix_dim']

            assert pix_dim % 2 == 0, "keep size of scene even so PSF is automatically odd"

            # define scene dimensions to mimic airsas scene
            RP.define_scene_dimensions(scene_dim_x=[sys['scene_dim_x'][0], sys['scene_dim_x'][1]],  # meters
                                       scene_dim_y=[sys['scene_dim_y'][0], sys['scene_dim_y'][1]],  # meters
                                       scene_dim_z=[sys['scene_dim_z'][0], sys['scene_dim_z'][1]],  # set z to 0
                                       pix_dim_sim=[pix_dim, pix_dim, 1],  # define simulated and BF dimensions
                                       pix_dim_bf=[pix_dim, pix_dim, 1])

            # Crop waveform will scale num samples to fit scene size
            RP.generate_transmit_signal(crop_wfm=True)

            BF = Beamformer(RP=RP, interp='nearest', mp=False, r=100)

            psnr = self.sim_config[C.WAVEFORM_SNR].getfloat(C.PSNR)

            noise_val = 1/(10**(psnr/10))
            print("Noise val is", noise_val)

            for img_index, name in enumerate(image_names):
                print("Processing image", name)

                input_img = os.path.join(LOAD_IMG_DIR, name)
                gt_scatterers = load_img_and_preprocess(input_img, pix_dim)
                gt_scatterers = gt_scatterers.ravel()[RP.circle_indeces]
                gt_scatterers = normalize(gt_scatterers)

                save_sas_plot(c2g(gt_scatterers, RP.circle_indeces, pix_dim, pix_dim),
                        os.path.join(self.save_img_dir, 'gt_' + str(img_index) + '.png'))

                np.save(os.path.join(self.save_data_dir, 'gt' \
                                     + str(img_index) + '.npy'), c2g(gt_scatterers, RP.circle_indeces, pix_dim, pix_dim))

                gt_scatterers = torch.from_numpy(gt_scatterers)

                print("Simulating waveforms")
                wfms = delay_waveforms(RP, RP.pixels_3D_sim, gt_scatterers,
                                       noise=True, noise_std=noise_val, min_dist=RP.min_dist,
                                       scat_phase=None)

                print("Beamforming")
                complex_bf = BF.beamform(wfms, RP.pixels_3D_bf)
                complex_bf = complex_bf.detach().cpu().numpy()

                print("Saving data to", self.save_data_dir)
                np.save(os.path.join(self.save_data_dir, 'beamformed_scatterers_' + str(img_index) + '.npy'),
                        c2g(complex_bf, RP.circle_indeces, pix_dim, pix_dim))

                print("Saving image to", self.save_img_dir)
                save_sas_plot(c2g(np.absolute(complex_bf), RP.circle_indeces, pix_dim, pix_dim),
                        os.path.join(self.save_img_dir, 'beamformed_scatterers_' + str(img_index) + '.png'))
