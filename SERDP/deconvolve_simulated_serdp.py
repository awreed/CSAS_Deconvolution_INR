import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from sim_csas_package.utils import normalize
from deconv_methods.deconv_methods import DeconvMethods

var = 'deconv_data/depth_slice64'

scene = np.load(var + '_scene/scene.npy').squeeze()
psf = np.load(var + '_psf/psf3.npy').squeeze()

scene_norm = np.load(var + '_scene/scene_norm_counts.npy').squeeze()
psf_norm = np.load(var + '_psf/norm3.npy').squeeze()

gt = np.load(var + '_scene/voxel_amplitudes.npy')
row_start = 50
row_stop = 150
col_start = 3
col_stop = -3

das_scene = scene[row_start:row_stop, col_start:col_stop].squeeze()
psf = psf[row_start:row_stop, col_start:col_stop].squeeze()
#psf = psf[35:65, 20:60]
gt = gt[row_start:row_stop, col_start:col_stop].squeeze()

scene_norm = scene_norm[row_start:row_stop, col_start:col_stop].squeeze()

das_scene = das_scene / normalize(scene_norm)

psf_norm = psf_norm[row_start:row_stop, col_start:col_stop].squeeze()

indeces = [0, 12, 24, 35, 50, 62, 73, 84]

gt_scenes = []
psfs = []
das_scenes = []

# Larger window values actually give a worse quant error
W = 10

to_be_deconvolved = []

for i in range(0, len(indeces)):
    tmp_psf = np.load(var + '_psf/psf' + str(i) + '.npy').squeeze()
    tmp_psf = tmp_psf[row_start:row_stop, col_start:col_stop].squeeze()

    # last one
    if i == len(indeces) - 1:
        tmp = gt[:, indeces[i]-W:]
        tmp_psf = tmp_psf[:, indeces[i]:]
        tmp_das = das_scene[:, indeces[i]-W:]
    # first one
    elif i == 0:
        tmp = gt[:, indeces[i]:indeces[i + 1]+W]
        tmp_das = das_scene[:, indeces[i]:indeces[i + 1]+W]
        tmp_psf = tmp_psf[:, indeces[i]:indeces[i + 1]]
    else:
        tmp = gt[:, indeces[i]-W:indeces[i+1]+W]
        tmp_das = das_scene[:, indeces[i]-W:indeces[i+1]+W]
        tmp_psf = tmp_psf[:, indeces[i]:indeces[i+1]]

    tmp_psf = tmp_psf[1:, ...]
    print(tmp_psf.shape)

    gt_scenes.append(tmp)
    psfs.append(tmp_psf)
    das_scenes.append(tmp_das)

    to_be_deconvolved.append({
        'scene': tmp_das,
        'psf': tmp_psf,
        'gt': tmp
    })

DM = DeconvMethods(deconv_config='deconv.ini',
                   to_be_deconvolved=to_be_deconvolved,
                   deconv_dir='deconv_dir',
                   circular=False,
                   device='cuda:0')

DM.run_all_methods()


