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
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        if torch.cuda.is_available():
            self.dev = 'cuda:0'
            print("Found gpu and using", self.dev)
        else:
            self.dev = 'cpu'
            print("Did not find gpu so using", self.dev)

    def run(self):
        with torch.no_grad():
            SAVE_IMG_DIR = self.config[C.IO][C.SAVE_IMG_DIR]
            SAVE_DIR = self.config[C.IO][C.SAVE_DATA_DIR]
            LOAD_IMG_DIR = self.config[C.IO][C.LOAD_IMG_DIR]

            image_names = self.config[C.IO][C.LOAD_IMG_NAMES]
            image_names = image_names.split(',')
            image_names = [num.strip() for num in image_names]

            RP = RenderParameters(device=self.dev)

            theta_start = self.config[C.SAS_GEOMETRY].getint(C.THETA_START)
            theta_stop = self.config[C.SAS_GEOMETRY].getint(C.THETA_STOP)
            theta_step = self.config[C.SAS_GEOMETRY].getint(C.THETA_STEP)
            radius = self.config[C.SAS_GEOMETRY].getfloat(C.R)
            Z_TX = self.config[C.SAS_GEOMETRY].getfloat(C.Z_TX)
            Z_RX = self.config[C.SAS_GEOMETRY].getfloat(C.Z_RX)

            # define transducer positions relative to the scene
            RP.define_transducer_pos(theta_start=theta_start, theta_stop=theta_stop,
                                     theta_step=theta_step, r=radius, z_TX=Z_TX, z_RX=Z_RX)

            scene_dim_x = self.config[C.SCENE_DIMENSIONS][C.DIM_X]
            scene_dim_x = scene_dim_x.split(',')
            scene_dim_x = [float(num.strip()) for num in scene_dim_x]

            scene_dim_y = self.config[C.SCENE_DIMENSIONS][C.DIM_Y]
            scene_dim_y = scene_dim_y.split(',')
            scene_dim_y = [float(num.strip()) for num in scene_dim_y]

            scene_dim_z = self.config[C.SCENE_DIMENSIONS][C.DIM_Z]
            scene_dim_z = scene_dim_z.split(',')
            scene_dim_z = [float(num.strip()) for num in scene_dim_z]

            pix_dim = self.config[C.SCENE_DIMENSIONS].getint(C.PIX_DIM)

            # define scene dimensions to mimic airsas scene
            RP.define_scene_dimensions(scene_dim_x=[scene_dim_x[0], scene_dim_x[1]],  # meters
                                       scene_dim_y=[scene_dim_y[0], scene_dim_y[1]],  # meters
                                       scene_dim_z=[scene_dim_z[0], scene_dim_z[1]],  # set z to 0
                                       pix_dim_sim=[pix_dim, pix_dim, 1],  # define simulated and BF dimensions
                                       pix_dim_bf=[pix_dim, pix_dim, 1])

            # Crop waveform will scale num samples to fit scene size
            RP.generate_transmit_signal(crop_wfm=True)

            r_interp = self.config[C.BEAMFORMER].getint(C.r)

            BF = Beamformer(RP=RP, interp='nearest', mp=False, r=r_interp)

            psnr = self.config[C.WAVEFORM_SNR].getfloat(C.PSNR)

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
                        os.path.join(SAVE_IMG_DIR, 'gt_' + str(img_index) + '.png'))

                np.save(os.path.join(SAVE_IMG_DIR, 'gt' \
                                     + str(img_index) + '.npy'), gt_scatterers)

                gt_scatterers = torch.from_numpy(gt_scatterers)

                print("Simulating waveforms")
                wfms = delay_waveforms(RP, RP.pixels_3D_sim, gt_scatterers,
                                       noise=True, noise_std=noise_val, min_dist=RP.min_dist,
                                       scat_phase=None)

                print("Beamforming")
                complex_bf = BF.beamform(wfms, RP.pixels_3D_bf)
                complex_bf = complex_bf.detach().cpu().numpy()


                print("Saving result")
                np.save(os.path.join(SAVE_DIR, 'beamformed_scatterers_' + str(img_index) + '.npy'), complex_bf)

                imwrite(c2g(np.absolute(complex_bf), RP.circle_indeces, pix_dim),
                        os.path.join(SAVE_IMG_DIR, 'beamformed_scatterers_' + str(img_index) + '.png'))
