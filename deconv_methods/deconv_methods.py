import torch
import configparser
import constants.constants as C
from deconv_methods.inr_recon import INRRecon
from functools import partial
import os
from deconv_methods.grad_desc_recon import GradDescRecon
import numpy as np
from deconv_methods.wiener_filter import WienerDeconv
from deconv_methods.bremen_alg import BremenAlg
from deconv_methods.dip_recon import DIPRecon
from sim_csas_package.utils import save_sas_plot


class DeconvMethods:
    def  __init__(self, deconv_config, to_be_deconvolved, deconv_dir, device=None, circular=False):
        self.all_methods = [C.INR, C.GD, C.GD_TV, C.GD_GRAD_REG, C.DIP, C.WIENER, C.BREMEN]
        self.use_methods = []

        self.deconv_config = configparser.ConfigParser()
        self.deconv_config.read(deconv_config)

        if torch.cuda.is_available():
            self.dev = 'cuda:0'
        else:
            self.dev = 'cpu'
            print("Did not find gpu so using", self.dev)

        self.to_be_deconvolved = to_be_deconvolved
        self.deconv_dir = deconv_dir

        if device is None:
            self.device = 'cuda:0' if torch.cuda.isavailable() else 'cpu'
        else:
            self.device = device

        self.circular = circular

        self.process_deconv_config()


    # Setup a method to be used for deconvolution
    def add_method(self, name):
        assert name in self.all_methods
        if name == C.INR:
            INR = INRRecon(self.device, self.circular)
            kappa = self.deconv_config[name].getfloat(C.KAPPA)
            nf = self.deconv_config[name].getint(C.NF)
            lr = self.deconv_config[name].getfloat(C.LR)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)

            meth = {'name':name,
                    'func': partial(INR.recon, kappa, nf, lr, max_iter, save_every)}
            self.use_methods.append(meth)

        elif name == C.GD:
            GD = GradDescRecon(self.device, self.circular)
            lr = self.deconv_config[name].getfloat(C.LR)
            momentum = self.deconv_config[name].getfloat(C.MOMENTUM)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)

            meth = {'name': name,
                    'func': partial(GD.recon, lr, momentum, 'none', 0., max_iter, save_every)}
            self.use_methods.append(meth)

        elif name == C.GD_TV or name == C.GD_GRAD_REG:
            GD = GradDescRecon(self.device, self.circular)
            lr = self.deconv_config[name].getfloat(C.LR)
            momentum = self.deconv_config[name].getfloat(C.MOMENTUM)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)
            reg_weight = self.deconv_config[name].getfloat(C.REG_WEIGHT)
            reg = self.deconv_config[name][C.REG]

            meth = {'name': name,
                    'func': partial(GD.recon, lr, momentum, reg, reg_weight, max_iter, save_every)}
            self.use_methods.append(meth)

        elif name == C.DIP:
            DIP = DIPRecon(self.device, self.circular)
            lr = self.deconv_config[name].getfloat(C.LR)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)

            meth = {'name': name,
                    'func': partial(DIP.recon, lr, max_iter, save_every)}
            self.use_methods.append(meth)

        elif name == C.WIENER:
            Wiener = WienerDeconv(self.device, self.circular)
            log_min = self.deconv_config[name].getfloat(C.MIN_LOG)
            log_max = self.deconv_config[name].getfloat(C.MAX_LOG)
            num_log_space = self.deconv_config[name].getint(C.NUM_LOG_SPACE)

            meth = {'name': name,
                    'func': partial(Wiener.recon, log_min, log_max, num_log_space)}
            self.use_methods.append(meth)

        elif name == C.BREMEN:
            Bremen = BremenAlg(self.device, self.circular)
            max_iter = self.deconv_config[name].getint(C.MAX_ITER)
            save_every = self.deconv_config[name].getint(C.SAVE_EVERY)

            meth = {'name': name,
                    'func': partial(Bremen.recon, max_iter, save_every)}
            self.use_methods.append(meth)

    # Add all the deconv methods from the config file
    def process_deconv_config(self):
        for key in self.deconv_config.keys():
            if key in self.all_methods:
                if key == 'DEFAULT':
                    continue
                print("Adding", key)
                self.add_method(key)
            else:
                if not key == 'DEFAULT':
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
            if task['gt'] is not None:
                assert task['gt'].ndim == 2, "Ground truth image input should be two dimensions (H, W)"

            if task['gt'] is not None:
                assert task['scene'].shape == task['gt'].shape, "Provided ground truth image should be same" \
                                                                " dimensions as DAS reconstructed image"
            if self.circular:
                assert task['scene'].shape[0] == task['scene'].shape[1], "DAS Input must be square (H == W) " \
                                                                         "to perform circular crop"
                if task['gt'] is not None:
                    assert task['gt'].shape[0] == task['gt'].shape[1], "GT must be square (H == W) " \
                                                                   "to perform circular crop"
            # TODO figure out how to check for generic complex type
            if task['scene'].dtype == np.complex64:
                task['scene'] = task['scene'].astype(np.complex128)

            if task['psf'].dtype == np.complex64:
                task['psf'] = task['psf'].astype(np.complex128)

            # loop over all methods to deconvolve with
            for meth in self.use_methods:
                meth_save_dir = os.path.join(save_dir, meth['name'])
                os.makedirs(meth_save_dir, exist_ok=True)
                print("Running", meth['name'], "on image", str(i))

                images, psnrs, ssims, lpips = meth['func'](task['scene'], task['psf'], task['gt'], meth_save_dir)

                save_sas_plot(task['gt'], os.path.join(meth_save_dir, 'gt.png'))
                save_sas_plot(np.abs(task['scene']), os.path.join(meth_save_dir, 'gt_das.png'))
                save_sas_plot(np.abs(task['psf']), os.path.join(meth_save_dir, 'psf.png'))

                np.save(os.path.join(meth_save_dir, 'final_deconv.npy'), images[-1])
                np.save(os.path.join(meth_save_dir, 'psnrs.npy'), np.asarray(psnrs))
                np.save(os.path.join(meth_save_dir, 'ssims.npy'), np.asarray(ssims))
                np.save(os.path.join(meth_save_dir, 'lpips.npy'), np.asarray(lpips))
