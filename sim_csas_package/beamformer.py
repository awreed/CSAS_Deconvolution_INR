import numpy as np
import torch
import torch.multiprocessing as mp
from sim_csas_package.utils import interpfft

class Beamformer:
    def __init__(self, RP, **kwargs):
        self.RP = RP

        self.trans = self.RP.trans

        self.interp = kwargs.get('interp', 'sinc')
        assert self.interp in ['sinc', 'nearest'], "interp should be either \
            sinc or nearest"
        self.dev = self.RP.dev
        self.scene = None
        self.mp = kwargs.get('mp', True)
        self.r = kwargs.get('r', 1.0)
        if self.interp == 'sinc':
          print("Setting r = 1 if using sinc for beamforming")
          self.r = 1

        torch.multiprocessing.set_start_method('spawn', force=True)

    def beamform(self, wfms, pixels, group_delay=0, temps=None):
        print("Beamforming")
        self.wfms = wfms.detach().cpu()
        num_wfms = self.wfms.shape[0]
      
        # if using nearest, then upsample waveforms 
        if self.interp == 'nearest':
          wfms_up = []
          for i in range(0, num_wfms):
            wfm = self.wfms[i, ...]
            wfms_up.append(interpfft(wfm, self.r))

          self.wfms = torch.from_numpy(np.array(wfms_up))

        self.pixels = pixels.detach().cpu()
        self.group_delay = group_delay * self.r
        self.temps = temps

        if self.mp:
            scene = self.beamform_mp()
        else:
            scene = self.beamform_no_mp()

        return scene
        
    
    def bf_trans(self, index):
        dist1 = torch.sqrt(torch.sum((self.pixels - \
        self.trans[index].tx_pos.unsqueeze(0).detach().cpu()) ** 2, 1))
        dist2 = torch.sqrt(torch.sum((self.pixels - \
            self.trans[index].rx_pos.unsqueeze(0).detach().cpu()) ** 2, 1))
       
        # Check if per ping temperature provided
        if not self.temps:
          tof = (((dist1 + dist2) - self.RP.min_dist) / self.RP.c) * self.RP.Fs * self.r
        else:
          # compute speed of sound from temp
          c = 331.4 + 0.6*self.temps[index]
          tof = (((dist1 + dist2) - self.RP.min_dist) / c) * self.RP.Fs * self.r

        tof = tof.detach().cpu().numpy()
        tof = tof + self.group_delay
        
        if self.interp == 'sinc':
            sinc_window = [np.sinc(tof - i) for i in \
                range(0, self.RP.num_samples)]

            sinc_window = np.array(sinc_window)
            sinc_window = sinc_window.T
            sinc_window = torch.from_numpy(sinc_window).to(self.wfms.device)

            scene = torch.sum(self.wfms[index, :][None, :] * sinc_window, dim=1)
        else:
            tof = tof.astype(int)
            scene = self.wfms[index, tof]
                 
        return scene

    # multiproceessing beamformer
    def beamform_mp(self):
        num_workers = mp.cpu_count()
        pool = mp.Pool(5)
        # put each transduer on diff thread
        scene_list = pool.map(self.bf_trans, range(len(self.trans)))

        # stack list and sum
        scene_vector = torch.stack((scene_list))
        scene = torch.sum(scene_vector, dim=0)
        return scene
        
    def beamform_no_mp(self):

        scene = torch.zeros(len(self.RP.circle_indeces)).to(self.wfms.device)
        T_wfm = 1 / self.RP.Fs

        for i, trans in enumerate(self.trans):
            #pdb.set_trace()
            dist1 = torch.sqrt(torch.sum((self.pixels - \
                trans.tx_pos.unsqueeze(0).detach().cpu()) ** 2, 1))
            dist2 = torch.sqrt(torch.sum((self.pixels - \
                trans.rx_pos.unsqueeze(0).detach().cpu()) ** 2, 1))
            
            # Check if per ping temperature provided
            if not self.temps:
              tof = (((dist1 + dist2) - self.RP.min_dist) / self.RP.c) * self.RP.Fs * self.r
            else:
              # from physics
              c = 331.4 + 0.6*self.temps[i]
              tof = (((dist1 + dist2) - self.RP.min_dist) / c) * self.RP.Fs * self.r

            tof = tof.detach().cpu().numpy()
            tof = tof + self.group_delay

            if self.interp == 'sinc':
                sinc_window = [np.sinc((tof - i)) for i in \
                    range(0, self.RP.num_samples)]

                sinc_window = np.array(sinc_window)
                sinc_window = sinc_window.T
                sinc_window = torch.from_numpy(sinc_window).to(self.wfms.device)

                scene = scene + \
                    torch.sum(self.wfms[i, :][None, :] * sinc_window, dim=1)
            else:
                tof = tof.astype(int)
                scene = scene + self.wfms[i, tof]

        return scene
   
