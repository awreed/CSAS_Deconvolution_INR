import torch
from sim_csas_package.render_parameters import RenderParameters
import configparser
from sim_csas_package.utils import process_sys_config
import constants.constants as C
from deconv_methods.inr_recon import INRRecon
from functools import partial
import os
from deconv_methods.grad_desc_recon import GradDescRecon
import numpy as np
from deconv_methods.wiener_filter import WienerDeconv
from deconv_methods.bremen_alg import BremenAlg
from deconv_methods.dip_recon import DIPRecon


class DeconvMethods:
    def __init__(self, deconv_config, sys_config, to_be_deconvolved, deconv_dir):
        self.all_methods = [C.INR, C.GD, C.GD_TV, C.GD_GRAD_REG, C.DIP, C.WIENER, C.BREMEN]
        self.use_methods = []

        self.sys_config = configparser.ConfigParser()
        self.sys_config.read(sys_config)

        self.deconv_config = configparser.ConfigParser()
        self.deconv_config.read(deconv_config)

        if torch.cuda.is_available():
            self.dev = 'cuda:0'
        else:
            self.dev = 'cpu'
            print("Did not find gpu so using", self.dev)

        with torch.no_grad():
            # Process the system config init file
            sys = process_sys_config(self.sys_config)

            self.RP = RenderParameters(device=self.dev, Fs=sys['fs'], c=sys['c'],
                                  f_start=sys['f_start'], f_stop=sys['f_stop'],
                                  t_start=sys['t_start'], t_stop=sys['t_stop'],
                                  win_ratio=sys['win_ratio'])

            # define transducer positions relative to the scene
            self.RP.define_transducer_pos(theta_start=sys['theta_start'], theta_stop=sys['theta_stop'],
                                     theta_step=sys['theta_step'], r=sys['radius'], z_TX=sys['Z_TX'],
                                     z_RX=sys['Z_RX'])

            pix_dim = sys['pix_dim']
            assert pix_dim % 2 == 0, "keep size of scene even so PSF is automatically odd"

            # define scene dimensions to mimic airsas scene
            self.RP.define_scene_dimensions(scene_dim_x=[sys['scene_dim_x'][0], sys['scene_dim_x'][1]],  # meters
                                       scene_dim_y=[sys['scene_dim_y'][0], sys['scene_dim_y'][1]],  # meters
                                       scene_dim_z=[sys['scene_dim_z'][0], sys['scene_dim_z'][1]],  # set z to 0
                                       pix_dim_sim=[pix_dim, pix_dim, 1],  # define simulated and BF dimensions
                                       pix_dim_bf=[pix_dim, pix_dim, 1])

        self.to_be_deconvolved = to_be_deconvolved
        self.deconv_dir = deconv_dir
        self.process_deconv_config()

    # Setup a method to be used for deconvolution
    def add_method(self, name):
        assert name in self.all_methods
        if name == C.INR:
            INR = INRRecon(self.RP)
            kappa = self.deconv_config[name].getfloat(C.KAPPA)
            nf = self.deconv_config[name].getint(C.NF)
            lr = self.deconv_config[name].getfloat(C.LR)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)

            meth = {'name':name,
                    'func': partial(INR.recon, kappa, nf, lr, max_iter, save_every)}
            self.use_methods.append(meth)

        elif name == C.GD:
            GD = GradDescRecon(self.RP)
            lr = self.deconv_config[name].getfloat(C.LR)
            momentum = self.deconv_config[name].getfloat(C.MOMENTUM)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)

            meth = {'name': name,
                    'func': partial(GD.recon, lr, momentum, 'none', 0., max_iter, save_every)}
            self.use_methods.append(meth)

        elif name == C.GD_TV or name == C.GD_GRAD_REG:
            GD = GradDescRecon(self.RP)
            lr = self.deconv_config[name].getfloat(C.LR)
            momentum = self.deconv_config[name].getfloat(C.MOMENTUM)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)
            reg_weight = self.deconv_config[name].getyfloat(C.REG_WEIGHT)
            reg = self.deconv_config[name][C.REG]

            meth = {'name': name,
                    'func': partial(GD.recon, lr, momentum, reg, reg_weight, max_iter, save_every)}
            self.use_methods.append(meth)


    # Add all the deconv methods from the config file
    def process_deconv_config(self):
        for key in self.deconv_config.keys():
            if key in self.all_methods:
                if self.deconv_config[key].getboolean(C.USE):
                    self.add_method(key)
            else:
                print("Method name not recognized in .ini file. User provided", key,
                      "but possible methods are", self.all_methods)


    # Run all the deconv methods on the provided data
    def run_all_methods(self):
        # Loop over all scenes to be deconvolved
        for i, task in enumerate(self.to_be_deconvolved):
            save_dir = os.path.join(self.deconv_dir, 'image' + str(i))
            os.makedirs(save_dir, exist_ok=True)

            assert task['scene'].ndim == 2, "DAS (scene) input should be two dimensions (H, W)"
            assert task['psf'].ndim == 2, "PSF input should be two dimensions (H-1, W-1)"
            assert task['gt'].ndim == 2, "Ground truth image input should be two dimensions (H, W)"

            if task['gt'] is not None:
                assert task['scene'].shape == task['gt'].shape, "Provided ground truth image should be same dimensions as " \
                                                                "DAS reconstructed image"

            # Deconv methods expect DAS vector to be 1D
            task['scene'] = task['scene'].ravel()[self.RP.circle_indeces]

            # loop over all methods to deconvolve with
            for meth in self.use_methods:
                images, psnrs, ssims, lpips = meth['func'](task['scene'], task['psf'], task['gt'], save_dir)
                np.save(os.path.join(save_dir, 'final_deconv.npy'), images[-1])
                np.save(os.path.join(save_dir, 'psnrs.npy'), np.asarray(psnrs))
                np.save(os.path.join(save_dir, 'ssims.npy'), np.asarray(ssims))
                np.save(os.path.join(save_dir, 'lpips.npy'), np.asarray(lpips))
