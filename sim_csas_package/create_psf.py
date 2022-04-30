import torch
import os
from sim_csas_package.render_parameters import RenderParameters
from sim_csas_package.utils import save_sas_plot, c2g
from sim_csas_package.beamformer import Beamformer
from sim_csas_package.waveform_processing import delay_waveforms
import configparser
import numpy as np
from sim_csas_package.utils import process_sys_config


class CreatePSF:
    def __init__(self, sys_config, save_img_dir, save_data_dir):
        self.sys_config = configparser.ConfigParser()
        self.sys_config.read(sys_config)

        self.save_img_dir = save_img_dir
        self.save_data_dir = save_data_dir

        if torch.cuda.is_available():
            self.dev = 'cuda:0'
        else:
            self.dev = 'cpu'
            print("Did not find gpu so using", self.dev)

    def run(self):
        with torch.no_grad():
            # Process the system config init file
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
            pix_dim = pix_dim - 1

            assert pix_dim % 2 == 1, "Pix dimension should be even so that PSF shape is odd"

            # define scene dimensions to mimic airsas scene
            RP.define_scene_dimensions(scene_dim_x=[sys['scene_dim_x'][0], sys['scene_dim_x'][1]],  # meters
                                       scene_dim_y=[sys['scene_dim_y'][0], sys['scene_dim_y'][1]],  # meters
                                       scene_dim_z=[sys['scene_dim_z'][0], sys['scene_dim_z'][1]],  # set z to 0
                                       pix_dim_sim=[pix_dim, pix_dim, 1],  # define simulated and BF dimensions
                                       pix_dim_bf=[pix_dim, pix_dim, 1])

            # Crop waveform will scale num samples to fit scene size
            RP.generate_transmit_signal(crop_wfm=True)

            BF = Beamformer(RP=RP, interp='nearest', mp=False, r=100)

            single_scatterer = torch.zeros((pix_dim, pix_dim))
            phase = torch.zeros((pix_dim, pix_dim))
            single_scatterer[int(pix_dim//2), int(pix_dim//2)] = 1

            single_scatterer = single_scatterer.view(-1)[RP.circle_indeces]
            phase = phase.view(-1)[RP.circle_indeces]

            print("Simulating waveforms")
            wfms = delay_waveforms(RP, RP.pixels_3D_sim, single_scatterer,
                                   noise=False, noise_std=0., min_dist=RP.min_dist,
                                   scat_phase=phase)

            print("Beamforming")
            complex_bf = BF.beamform(wfms, RP.pixels_3D_bf)
            complex_bf = complex_bf.detach().cpu().numpy()


            print("Saving data to", self.save_data_dir)
            np.save(os.path.join(self.save_data_dir, 'psf' + '.npy'), c2g(complex_bf, RP.circle_indeces, pix_dim, pix_dim))

            print("Saving image to", self.save_img_dir)
            save_sas_plot(c2g(np.absolute(complex_bf), RP.circle_indeces, pix_dim, pix_dim),
                    os.path.join(self.save_img_dir, 'psf' + '.png'))
