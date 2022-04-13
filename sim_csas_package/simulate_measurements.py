import torch
import os
import constants.constants as C
from sim_csas_package.render_parameters import RenderParameters
from sim_csas_package.utils import imwrite, normalize, load_img_and_preprocess, c2g
from sim_csas_package.beamformer import Beamformer
from sim_csas_package.waveform_processing import delay_waveforms
import configparser
import numpy as np


class SimulateMeasurements:
    def __init__(self, sim_config, sys_config, save_img_dir, save_data_dir):
        self.sim_config = configparser.ConfigParser()
        self.sim_config.read(sim_config)

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
            LOAD_IMG_DIR = self.sim_config[C.IO][C.LOAD_IMG_DIR]

            image_names = self.sim_config[C.IO][C.LOAD_IMG_NAMES]
            image_names = image_names.split(',')
            image_names = [num.strip() for num in image_names]

            f_start = self.sys_config[C.WAVEFORM].getint(C.F_START)
            f_stop = self.sys_config[C.WAVEFORM].getint(C.F_STOP)
            t_start = self.sys_config[C.WAVEFORM].getfloat(C.T_START)
            t_stop = self.sys_config[C.WAVEFORM].getfloat(C.T_STOP)
            win_ratio = self.sys_config[C.WAVEFORM].getfloat(C.TUKEY_RATIO)
            fs = self.sys_config[C.SIGNAL_PROCESSING].getint(C.FS)
            c = self.sys_config[C.SIGNAL_PROCESSING].getint(C.C)

            RP = RenderParameters(device=self.dev, Fs=fs, c=c,
                                  f_start=f_start, f_stop=f_stop,
                                  t_start=t_start, t_stop=t_stop,
                                  win_ratio=win_ratio)



            theta_start = self.sys_config[C.SAS_GEOMETRY].getint(C.THETA_START)
            theta_stop = self.sys_config[C.SAS_GEOMETRY].getint(C.THETA_STOP)
            theta_step = self.sys_config[C.SAS_GEOMETRY].getint(C.THETA_STEP)
            radius = self.sys_config[C.SAS_GEOMETRY].getfloat(C.R)
            Z_TX = self.sys_config[C.SAS_GEOMETRY].getfloat(C.Z_TX)
            Z_RX = self.sys_config[C.SAS_GEOMETRY].getfloat(C.Z_RX)

            # define transducer positions relative to the scene
            RP.define_transducer_pos(theta_start=theta_start, theta_stop=theta_stop,
                                     theta_step=theta_step, r=radius, z_TX=Z_TX, z_RX=Z_RX)

            scene_dim_x = self.sys_config[C.SCENE_DIMENSIONS][C.DIM_X]
            scene_dim_x = scene_dim_x.split(',')
            scene_dim_x = [float(num.strip()) for num in scene_dim_x]

            scene_dim_y = self.sys_config[C.SCENE_DIMENSIONS][C.DIM_Y]
            scene_dim_y = scene_dim_y.split(',')
            scene_dim_y = [float(num.strip()) for num in scene_dim_y]

            scene_dim_z = self.sys_config[C.SCENE_DIMENSIONS][C.DIM_Z]
            scene_dim_z = scene_dim_z.split(',')
            scene_dim_z = [float(num.strip()) for num in scene_dim_z]

            pix_dim = self.sys_config[C.SCENE_DIMENSIONS].getint(C.PIX_DIM)

            # define scene dimensions to mimic airsas scene
            RP.define_scene_dimensions(scene_dim_x=[scene_dim_x[0], scene_dim_x[1]],  # meters
                                       scene_dim_y=[scene_dim_y[0], scene_dim_y[1]],  # meters
                                       scene_dim_z=[scene_dim_z[0], scene_dim_z[1]],  # set z to 0
                                       pix_dim_sim=[pix_dim, pix_dim, 1],  # define simulated and BF dimensions
                                       pix_dim_bf=[pix_dim, pix_dim, 1])

            # Crop waveform will scale num samples to fit scene size
            RP.generate_transmit_signal(crop_wfm=True)

            BF = Beamformer(RP=RP, interp='nearest', mp=False, r=100)

            psnr = self.sim_config[C.WAVEFORM_SNR].getfloat(C.PSNR)

            noise_val = 1/(10**(psnr/10))
            print("Noise val is", noise_val)

            assert pix_dim % 2 == 0, "keep size of scene even so PSF is automatically odd"

            for img_index, name in enumerate(image_names):
                print("Processing image", name)

                input_img = os.path.join(LOAD_IMG_DIR, name)
                gt_scatterers = load_img_and_preprocess(input_img, pix_dim)
                gt_scatterers = gt_scatterers.ravel()[RP.circle_indeces]
                gt_scatterers = normalize(gt_scatterers)

                imwrite(c2g(gt_scatterers, RP.circle_indeces, pix_dim),
                        os.path.join(self.save_img_dir, 'gt_' + str(img_index) + '.png'))

                np.save(os.path.join(self.save_data_dir, 'gt' \
                                     + str(img_index) + '.npy'), gt_scatterers)

                gt_scatterers = torch.from_numpy(gt_scatterers)

                print("Simulating waveforms")
                wfms = delay_waveforms(RP, RP.pixels_3D_sim, gt_scatterers,
                                       noise=True, noise_std=noise_val, min_dist=RP.min_dist,
                                       scat_phase=None)

                print("Beamforming")
                complex_bf = BF.beamform(wfms, RP.pixels_3D_bf)
                complex_bf = complex_bf.detach().cpu().numpy()

                print("Saving data to", self.save_data_dir)
                np.save(os.path.join(self.save_data_dir, 'beamformed_scatterers_' + str(img_index) + '.npy'), complex_bf)

                print("Saving image to", self.save_img_dir)
                imwrite(c2g(np.absolute(complex_bf), RP.circle_indeces, pix_dim),
                        os.path.join(self.save_img_dir, 'beamformed_scatterers_' + str(img_index) + '.png'))
