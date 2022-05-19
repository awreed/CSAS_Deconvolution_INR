import torch
import numpy as np
from sim_csas_package.waveform_processing import hilbert_torch
import scipy.signal
import math
from sim_csas_package.transducer import Transducer

class RenderParameters:
    def __init__(self, **kwargs):
        self.Fs = kwargs.get('Fs', 100000) * 1.0
        self.num_samples = None # Computed under generateTransmitSignal()
        self.data = {}
        self.num_samples_transmit = None

        self.dev = kwargs.get('device', None)

        # Will be used to create torch constant, not differentiable at this time
        self.f_start = kwargs.get('f_start', 30000)  # Chirp start frequency
        self.f_stop = kwargs.get('f_stop', 10000)  # Chirp stop frequency
        self.t_start = kwargs.get('t_start', 0)  # Chirp start time
        self.t_stop = kwargs.get('t_stop', .001)  # Chirp stop time
        self.win_ratio = kwargs.get('win_ratio', 0.1)  # Tukey window ratio for chirp
        self.c = kwargs.get('c', 339.5)

        self.transmit_signal = None  # Transmitted signal (Set to torch type)
        self.pulse = None  # Hilbert transform of the transmitted signal
        self.fft_pulse = None  # FFT of the hilbert transform of transmitted signal
        self.scene = None  # Stores the processed .obj file
        self.theta_start = None  # Projector start angle in degrees
        self.theta_stop = None  # Projector stop angle in degrees
        self.theta_step = None  # Projector step angle in degrees
        self.num_thetas = None  # Projector number of theta positions
        self.num_proj = None  # Number of projectors
        self.projectors = None  # Array of projector 3D coordinates in meters (Set to torch type)
        self.thetas = None  # Array of theta values in degrees

        self.scene_dim_x = None # Ensonified scene dimensions [-x, x]
        self.scene_dim_y = None # Ensonified scene dimensions [-y, y]
        self.scene_dim_z = None # Ensonified scene dimensions [-z, z]
        self.pix_dim = None # N pixels in beamformed image in format [x, y, z]
        self.x_vect = None # Vector of scene pixels x positions
        self.y_vect = None # Vector of scene pixel y positions
        self.z_vect = None # Vector of scene pixel z positions
        self.z_TX = None
        self.Z_RX = None
        self.num_pix_3D = None
        self.scene_center = None # Center of the ensonified scene
        self.pix_pos = None
        self.hydros = None
        self.nf = 128
        self.pixels_3D_bf = None
        self.pixels_3D_sim = None
        self.perturb = False
        self.min_dist = 0.00

    def define_scene_dimensions(self, **kwargs):
        self.scene_dim_x = kwargs.get('scene_dim_x', None)
        self.scene_dim_y = kwargs.get('scene_dim_y', None)
        self.scene_dim_z = kwargs.get('scene_dim_z', None)

        # Beamform dimensions
        self.pix_dim_bf = kwargs.get('pix_dim_bf', None)

        assert abs(self.scene_dim_x[0]) == abs(self.scene_dim_x[1])
        assert abs(self.scene_dim_y[0]) == abs(self.scene_dim_y[1])

        x_vect_bf = np.linspace(self.scene_dim_x[0], self.scene_dim_x[1], 
            self.pix_dim_bf[0], endpoint=True)
        y_vect_bf = np.linspace(self.scene_dim_y[0], self.scene_dim_y[1],
             self.pix_dim_bf[1], endpoint=True)
        z_vect_bf = np.linspace(self.scene_dim_z[0], self.scene_dim_z[1],
             self.pix_dim_bf[2], endpoint=True)

        self.num_pix_3D_bf = np.size(x_vect_bf) * np.size(y_vect_bf) *\
             np.size(self.z_vect)

        self.scene_center = np.array([0., 0., 0.])

        (x, y, z) = np.meshgrid(x_vect_bf, y_vect_bf, z_vect_bf)

        pixel_grid = np.hstack((np.reshape(x, 
            (np.size(x), 1)), np.reshape(y, (np.size(y), 1)),
            np.reshape(z, (np.size(z), 1))))

        self.circle_indeces = np.where(pixel_grid[..., 0]**2 +\
            pixel_grid[...,1]**2 <= self.scene_dim_x[0]**2)

        self.mask = np.zeros((self.num_pix_3D_bf))
        self.mask[self.circle_indeces] = 1
        self.mask = self.mask.reshape(self.pix_dim_bf[0], self.pix_dim_bf[0])

        pixel_circle = pixel_grid[self.circle_indeces]

        self.pixels_3D_bf = torch.from_numpy(pixel_circle)
        self.pixels_3D_sim = self.pixels_3D_bf

    def generate_transmit_signal(self,wfm=None, **kwargs):

      crop_wfm = kwargs.get('crop_wfm', False)
      pixels_3D_sim = self.pixels_3D_sim.detach().cpu().numpy()

      # Find min and max time of flight to edges of scene
      if crop_wfm:
        edge_indeces = np.where(
          (pixels_3D_sim[:, 0] == min(pixels_3D_sim[:, 0])) |
          (pixels_3D_sim[:, 0] == max(pixels_3D_sim[:, 0])) |
          (pixels_3D_sim[:, 1] == min(pixels_3D_sim[:, 1])) |
          (pixels_3D_sim[:, 1] == max(pixels_3D_sim[:, 1])) 
        )

        edges = pixels_3D_sim[edge_indeces]

        min_dist = []
        max_dist = []

        for trans in self.trans:
          tx = trans.tx_pos.detach().cpu().numpy()[None, ...]
          rx = trans.rx_pos.detach().cpu().numpy()[None, ...]

          dist1 = np.sqrt(np.sum((edges - tx)**2, 1))
          dist2 = np.sqrt(np.sum((edges - rx)**2, 1))

          dist = dist1 + dist2

          min_dist.append(np.min(dist))
          max_dist.append(np.max(dist))
        
        # pad min and max distance by a bit
        self.min_dist = min(min_dist) - .05 # (m)
        self.max_dist = max(max_dist) + .05 # (m)


        assert self.max_dist > self.min_dist

        t_dur = (self.max_dist - self.min_dist) / self.c

        self.num_samples = math.ceil(t_dur * self.Fs)
      else:
        self.num_samples = 1000
        self.min_dist = 0.0

      if wfm is None:
          times = np.linspace(self.t_start, self.t_stop - 1 / self.Fs, num=int((self.t_stop - self.t_start) * self.Fs))
          LFM = scipy.signal.chirp(times, self.f_start, self.t_stop, self.f_stop)  # Generate LFM chirp

          tuk_win = scipy.signal.windows.tukey(len(times), self.win_ratio)
          LFM = LFM*tuk_win

          ind1 = 0  # Not supporting staring time other than zero atm
          ind2 = ind1 + len(LFM)

          #print("Num samples", self.num_samples)
          sig = np.full(int(self.num_samples), 1e-8)

          sig[ind1:ind2] = LFM  # Insert chirp into receive signal
      else:
          sig = np.full(int(self.num_samples), 1e-8)

          sig[:len(wfm)] = wfm.squeeze()
          LFM = wfm.squeeze()

      sig = torch.from_numpy(sig)

      self.pulse_fft_kernel = torch.fft.fft(hilbert_torch(sig))

      # Used to build received waveform
      LFM = torch.from_numpy(LFM)
      self.transmit_signal = LFM
      self.num_samples_transmit = len(LFM)


    def define_transducer_pos(self, **kwargs):
        self.theta_start = kwargs.get('theta_start', 0)
        self.theta_stop = kwargs.get('theta_stop', 359)
        self.theta_step = kwargs.get('theta_step', 1)
        self.r = kwargs.get('r', None)
        self.z_TX = kwargs.get('z_TX', None)
        self.z_RX = kwargs.get('z_RX', None)

        self.thetas = range(self.theta_start, self.theta_stop, self.theta_step)

        self.num_thetas = len(self.thetas)

        trans = []

        # Pack every projector position into an array
        for i in range(0, self.num_thetas):
            tx_pos = torch.tensor(
                [self.r * math.cos(np.deg2rad(self.thetas[i])), self.r * math.sin(np.deg2rad(self.thetas[i])),
                 self.z_TX])
            rx_pos = torch.tensor(
                [self.r * math.cos(np.deg2rad(self.thetas[i])), self.r * math.sin(np.deg2rad(self.thetas[i])),
                 self.z_RX])
            trans.append(Transducer(tx_pos=tx_pos, rx_pos=rx_pos))

        self.num_proj = self.num_thetas
        self.trans = trans

    
