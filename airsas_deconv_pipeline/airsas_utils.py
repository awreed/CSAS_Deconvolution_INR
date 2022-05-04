import numpy as np
import glob
import csv
import scipy.signal
import os
import sys
sys.path.append('../')
from sim_csas_package.render_parameters import RenderParameters
from sim_csas_package.beamformer import Beamformer
from sim_csas_package.waveform_processing import delay_waveforms
from sim_csas_package.utils import c2g
import torch

"""Method to remove the mean amplitude and phase response from time series"""
def remove_room(ts):
    # ts.shape = [360, 1000]
    ang = np.angle(ts)
    cm = np.mean(ts, 0)
    # cm.shape = [1000]
    dang = np.angle(np.exp(1j * (np.angle(cm[None, ...]) - np.angle(ts))))
    # dang.shape = [360, 1000]
    beta = 1. / (1 + np.abs(dang) ** 2)
    alpha = 1. / (1 + (np.abs(cm)[None, ...] - np.abs(ts)) ** 2)
    rm = np.abs(cm)[None, ...] * np.exp(1j * ang)
    nts = ts - alpha * beta * rm

    return nts

# Method for processing AirSAS data
def process_bf_files(direc):
    coord_file = 'Coordinates.csv'
    sys_file = 'SysParams.csv'
    wfm_params_file = 'WaveformParams.csv'
    wfm_file = 'Waveforms.csv'

    angles = []
    temps = []
    heights = []

    # Read the measurement angles
    with open(os.path.join(direc, coord_file)) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i > 0:
                angles.append(float(row[1]))
                temps.append(float(row[3]))
                heights.append(float(row[2]))
        # angles.pop(0)

    # super ineffecient way to check if all heights are the same
    heights = np.array(heights)
    for i in range(1, len(heights)):
        assert heights[i - 1] == heights[i]

    wfm = []

    with open(os.path.join(direc, wfm_file)) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            wfm.append(float(row[0]))

    # Read the system parameters
    TX = np.zeros((3))
    RX = np.zeros((3))
    GD = 0
    FS = 0
    Center = np.zeros((3))

    with open(os.path.join(direc, sys_file)) as fp:
        reader = csv.reader(fp)
        for row in reader:
            if row[0] == 'Speaker x':
                TX[0] = float(row[1])
            if row[0] == 'Speaker y':
                TX[1] = float(row[1])
            if row[0] == 'Speaker z':
                TX[2] = float(row[1])
            if row[0] == 'Mic1 x':
                RX[0] = float(row[1])
            if row[0] == 'Mic1 y':
                RX[1] = float(row[1])
            if row[0] == 'Mic1 z':
                RX[2] = (row[1])

            if row[0] == 'Group Delay':
                GD = float(row[1])

            if row[0] == 'Fs':
                FS = float(row[1])

            if row[0] == 'Center x':
                Center[0] = float(row[1])
            if row[0] == 'Center y':
                Center[1] = float(row[1])
            if row[0] == 'Center z':
                Center[2] = float(row[1])


    with open(os.path.join(direc, wfm_params_file)) as fp:
        reader = csv.reader(fp)
        for row in reader:
            wfm_params = row[0]
            break

    wfm_params = wfm_params.split('_')
    f_start = int(mat_str_2_float(wfm_params[1].replace('Hz', '')))
    f_stop = int(mat_str_2_float(wfm_params[2].replace('Hz', '')))
    t_dur = float(mat_str_2_float(wfm_params[3].replace('s', '')))
    win_ratio = float(mat_str_2_float(wfm_params[6]))

    files = glob.glob(os.path.join(direc, 'Flight*.csv'))
    num_flights = len(files)

    # count the number of samples
    num_samples = 0
    with open(files[0]) as fp:
        reader = csv.reader(fp)
        for row in reader:
            num_samples = num_samples + 1

    data = np.zeros((len(angles), num_samples))

    for angle, flight_num in zip(range(len(angles)), range(1, num_flights + 1)):
        file_name = "Flight-%06d.csv" % (flight_num)
        with open(os.path.join(direc, file_name)) as fp:
            reader = csv.reader(fp)
            for sample, row in enumerate(reader):
                data[angle, sample] = float(row[0])

    return TX, RX, Center, data, GD, wfm, temps, heights, f_start, f_stop, t_dur, win_ratio

def mat_str_2_float(s):
    s = s.split('E')
    base = float(s[0])

    pos=None
    if '+' in s[1]:
        pos=True
    elif '-' in s[1]:
        pos=False
    else:
        raise Exception('Could not parse waveform parameters from AirSAS. Check the parser logic.')
    if pos:
        exp = float(s[1].strip('+'))
        ans = base * 10 ** (exp)
    else:
        exp = float(s[1].strip('-'))
        ans = base * 10 ** (-exp)

    return ans

def compute_psf(scene_dir, device, SIZE_W, SIZE_H):
      TX, RX, center, data, GD, rec_wfm, temps, heights, f_start, f_stop, t_dur, win_ratio = process_bf_files(scene_dir)

      zTx = (TX[2] + heights[0]/1000) - center[2]
      zRx = (RX[2] + heights[0]/1000) - center[2]

      RP = RenderParameters(device=device)

      print("Assuming AirSAS Radius is 0.85 meters.")
      RP.define_transducer_pos(theta_start=90,
                               theta_stop=-270,
                               theta_step=-1,
                               r=0.85,
                               z_TX=zTx,
                               z_RX=zRx)

      RP.define_scene_dimensions(scene_dim_x=[-.3, .3],
                                 scene_dim_y=[-.3, .3],
                                 scene_dim_z=[0, 0],
                                 pix_dim_sim=[SIZE_W-1, SIZE_H-1, 1],
                                 pix_dim_bf=[SIZE_W-1, SIZE_H-1, 1],
                                )

      wfm = gen_real_lfm(RP.Fs, f_start, f_stop, t_dur, window=True, win_ratio=win_ratio)

      RP.generate_transmit_signal(wfm=wfm, crop_wfm=False)

      BF = Beamformer(RP=RP, interp='nearest', mp=False, r=100)

      wfms = delay_waveforms(RP, torch.from_numpy(np.array([0., 0., 0.])), torch.from_numpy(np.array([1.])),
                             noise=False, noise_std=0.,
                             scat_phase=None)

      complex_psf = BF.beamform(wfms, RP.pixels_3D_bf)
      complex_psf = complex_psf.detach().cpu().numpy()

      complex_psf = c2g(complex_psf, RP.circle_indeces, SIZE_W-1, SIZE_H-1)

      return wfms, complex_psf

"""Method to create LFM waveform
Fs: sample rate
n_samples: number of samples in padded lfm
f_start: LFM start frequency
f_stop: LFM stop frequency
t_dur: LFM duration
window: Option to apply Tukey Window
win_ratio: Tukey window ratio
"""
def gen_real_lfm(Fs, f_start, f_stop, t_dur, window=True, win_ratio=0.1):
    times = np.linspace(0, t_dur - 1/Fs, num=int((t_dur)*Fs))
    LFM = scipy.signal.chirp(times, f_start, t_dur, f_stop)

    if window:
        tuk_win = scipy.signal.windows.tukey(len(LFM), win_ratio)
        LFM = tuk_win*LFM

    return LFM