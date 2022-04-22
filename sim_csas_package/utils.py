import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import torch
import constants.constants as C

def process_sys_config(sys_config):
    A = {}

    A['f_start'] = sys_config[C.WAVEFORM].getint(C.F_START)
    A['f_stop'] = sys_config[C.WAVEFORM].getint(C.F_STOP)
    A['t_start'] = sys_config[C.WAVEFORM].getfloat(C.T_START)
    A['t_stop'] = sys_config[C.WAVEFORM].getfloat(C.T_STOP)
    A['win_ratio'] = sys_config[C.WAVEFORM].getfloat(C.TUKEY_RATIO)
    A['fs'] = sys_config[C.SIGNAL_PROCESSING].getint(C.FS)
    A['c'] = sys_config[C.SIGNAL_PROCESSING].getint(C.C)

    A['theta_start'] = sys_config[C.SAS_GEOMETRY].getint(C.THETA_START)
    A['theta_stop'] = sys_config[C.SAS_GEOMETRY].getint(C.THETA_STOP)
    A['theta_step'] = sys_config[C.SAS_GEOMETRY].getint(C.THETA_STEP)
    A['radius'] = sys_config[C.SAS_GEOMETRY].getfloat(C.R)
    A['Z_TX'] = sys_config[C.SAS_GEOMETRY].getfloat(C.Z_TX)
    A['Z_RX'] = sys_config[C.SAS_GEOMETRY].getfloat(C.Z_RX)

    A['scene_dim_x'] = sys_config[C.SCENE_DIMENSIONS][C.DIM_X]
    A['scene_dim_x'] = A['scene_dim_x'].split(',')
    A['scene_dim_x'] = [float(num.strip()) for num in A['scene_dim_x']]

    A['scene_dim_y'] = sys_config[C.SCENE_DIMENSIONS][C.DIM_Y]
    A['scene_dim_y'] = A['scene_dim_y'].split(',')
    A['scene_dim_y'] = [float(num.strip()) for num in A['scene_dim_y']]

    A['scene_dim_z'] = sys_config[C.SCENE_DIMENSIONS][C.DIM_Z]
    A['scene_dim_z'] = A['scene_dim_z'].split(',')
    A['scene_dim_z'] = [float(num.strip()) for num in A['scene_dim_z']]

    A['pix_dim'] = sys_config[C.SCENE_DIMENSIONS].getint(C.PIX_DIM)

    return A

def g2c(grid):
  grid = grid.squeeze()
  x = np.linspace(-1, 1, grid.shape[0], endpoint=True)
  xy = np.meshgrid(x, x)
  xy = np.stack(xy, axis=-1)
  mask = np.zeros((grid.shape[0], grid.shape[0]))
  indeces = np.where(xy[..., 0]**2 + xy[..., 1]**2 <= 1)
  mask[indeces] = 1
  masked_grid = mask*grid

  return masked_grid, indeces

def c2g(circle, indeces, size):
  dtype = circle.dtype
  img = np.zeros(size*size, dtype=dtype)
  img[indeces] = circle
  img = img.reshape(size, size)
  return img



def save_sas_plot(img, path, x_size=0.2, y_size=0.2, log=False): 
  plt.clf()
  if log:
    plt.imshow(20*np.log10(img), extent=[-x_size, x_size, -y_size, y_size],
        origin='lower')
    plt.colorbar(label='dB')
    plt.clim(vmin=-30, vmax=0) 
  else:
    plt.imshow(img, extent=[-y_size, y_size, -y_size, y_size])
    plt.colorbar()

  ax = plt.gca()
  ticks = np.around(np.arange(-x_size, x_size, .04), decimals=2)
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  plt.xlabel('m')
  plt.ylabel('m')
  plt.show()
  plt.savefig(path)


# Crops the PSF down to smaller size based off desired amount of energy
# (thresh)
def crop_psf(RP, psf, thresh):

  assert thresh > 0
  assert thresh < 1.0

  SIZE, _ = psf.shape
  row = psf[SIZE//2, SIZE//2:]

  total = np.sum(np.absolute(row))

  check = 0.00
  count = 0
  while(check/total < thresh):
    chunk = row[:count]
    check = np.sum(np.absolute(chunk))
    count = count + 1
    if count == row.shape[0]:
      return psf

  total_length = count

  cropped_psf = psf[SIZE//2-total_length:SIZE//2+total_length+1,
      SIZE//2-total_length:SIZE//2+total_length+1]

  #print("New PSF shape", cropped_psf.shape)

  return cropped_psf

def interpfft(x, r):
  nx = len(x)
  X = np.fft.fft(x)

  Xint = np.zeros(len(X)*r, dtype=np.complex128)
  nxint = len(Xint)

  if len(x) % 2 == 0:
    Xint[0:nx//2] = X[0:nx//2]
    Xint[nx//2] = X[nx//2]/2
    Xint[nxint-nx//2] = X[nx//2]/2
    Xint[nxint-nx//2+1:] = X[nx//2+1:]
  else:
    Xint[0:math.floor(nx/2)+1] = X[:math.floor(nx/2)+1]
    Xint[nxint-math.floor(nx/2):] = X[math.floor(nx/2)+1:]

  xint = np.fft.ifft(Xint)*r
  return xint

def drc(img, med, des_med):
  fp = (des_med - med * des_med)/(med - med * des_med)
  return (img*fp)/(fp*img-img+1)


def process_bf_files(direc):
    coord_file = 'Coordinates.csv'
    sys_file = 'SysParams.csv'
    wfm_params_file = 'WaveformParams.csv'
    wfm_file = 'Waveforms.csv'

    angles = []

    # Read the measurement angles
    with open(os.path.join(direc, coord_file)) as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i > 0:
                angles.append(float(row[1]))
        #angles.pop(0)
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

    #print(TX, RX, GD, FS, Center)

    with open(os.path.join(direc, wfm_params_file)) as fp:
        reader = csv.reader(fp)
        for row in reader:
            print(row)

    files = glob.glob(os.path.join(direc, 'Flight*.csv'))
    num_flights = len(files)

    # count the number of samples
    num_samples = 0
    with open(files[0]) as fp:
        reader = csv.reader(fp)
        for row in reader:
            num_samples = num_samples + 1

    data = np.zeros((len(angles), num_samples))

    for angle, flight_num in zip(range(len(angles)), range(1, num_flights+1)):
        file_name = "Flight-%06d.csv" % (flight_num)
        with open(os.path.join(direc, file_name)) as fp:
            reader = csv.reader(fp)
            for sample, row in enumerate(reader):
                data[angle, sample] = float(row[0])

    return TX, RX, Center, data, GD, wfm

def load_img_and_preprocess(path, SIZE, rotate=False):
    img = cv2.imread(path)                             
    if img.ndim == 3:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                                 
    img = cv2.resize(img, (SIZE, SIZE))  
    
    if rotate:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        
    return img
    
def normalize(data):                                                            
    return (data - data.min())/(data.max() - data.min() + 1e-9)                        

def TV(x):
  h, w = x.shape
  loss = torch.sum(torch.sum(torch.abs(x[1:, :] - x[:-1, :])) + torch.sum(torch.abs(x[:,
    1:] - x[:, :-1])))
  return loss

def L1_reg(x):
  h, w = x.shape
  loss = torch.sum(torch.abs(x))
  return loss

def L2_reg(x):
  h, w = x.shape
  loss = torch.sum(x**2)
  return loss

def grad_reg(x):
  if x.dim() == 2:
    x = x[None, None, ...]

  kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0,
    -1]]).double().to(x.device)[None, None, ...]
  ky = torch.tensor([[1, 2 ,1], [0, 0, 0], [-1, -2,
    -1]]).double().to(x.device)[None, None, ...]

  eps = 1e-7

  x_d = torch.nn.functional.conv2d(x, kx, padding=1)
  y_d = torch.nn.functional.conv2d(x, ky, padding=1)

  xy_d = torch.sqrt(x_d**2 + y_d**2 + eps).squeeze()

  return torch.sum(xy_d)

def imwrite(img, path):
  img = (normalize(img)*255).astype('uint8')
  img = Image.fromarray(img)
  img.save(path)

