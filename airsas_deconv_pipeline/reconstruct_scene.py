import sys
import numpy as np
sys.path.append('../')
from sim_csas_package.render_parameters import RenderParameters
from sim_csas_package.beamformer import Beamformer
from sim_csas_package.waveform_processing import replica_correlate_torch
from sim_csas_package.utils import c2g, save_sas_plot
from airsas_utils import remove_room, process_bf_files, gen_real_lfm, compute_psf
import torch
import os
from deconv_methods.deconv_methods import DeconvMethods

if __name__ == '__main__':
    SIZE_W, SIZE_H = 400, 400
    # AirSAS measurements to be reconstructed
    scene_dir = 'airsas_data/small_features_cutout_20k'
    # AirSAS measurements of empty scene
    bg_dir = 'airsas_data/bg_20k'
    # Deconv config file
    deconv_config = 'deconv.ini'
    save_directory = '20k_scene'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ########### DAS Reconstruct the scene #####################

    TX, RX, center, data, GD, rec_wfm, temps, heights, f_start, f_stop, t_dur, win_ratio = process_bf_files(scene_dir)

    # Get empty scene measurements
    if bg_dir is not None:
        _, _, _, blank_data, _, _, _, _, _, _, _, _ \
            = process_bf_files(bg_dir)
        # subtract the blank scene waveforms
        data = data - blank_data

    # remove the mean room response from data
    data = remove_room(data)

    # Height of TX/RX
    zTx = (TX[2] + heights[0] / 1000) - center[2]
    zRx = (RX[2] + heights[0] / 1000) - center[2]

    RP = RenderParameters(device=device)

    print("Assuming AirSAS Radius is 0.85 meters.")
    RP.define_transducer_pos(theta_start=90, theta_stop=-270,
                             theta_step=-1,
                             r=0.85, z_TX=zTx, z_RX=zRx)

    RP.define_scene_dimensions(scene_dim_x=[-.3, .3],
                               scene_dim_y=[-.3, .3],
                               scene_dim_z=[0, 0],
                               pix_dim_sim=[SIZE_W, SIZE_H, 1],
                               pix_dim_bf=[SIZE_W, SIZE_H, 1])

    wfm = gen_real_lfm(RP.Fs, f_start, f_stop, t_dur, window=True, win_ratio=win_ratio)

    RP.generate_transmit_signal(wfm=wfm, crop_wfm=False)

    (num_angles, num_samples) = data.shape

    # Match-filter the data with the transmit waveform
    dataRC = torch.zeros((num_angles, 1000), dtype=torch.complex128)
    for i in range(0, num_angles):
        data_tmp = torch.from_numpy(data[i, ...])
        dataRC[i, ...] = replica_correlate_torch(data_tmp, RP.pulse_fft_kernel)

    BF = Beamformer(RP=RP, interp='nearest', mp=True, r=100)

    with torch.no_grad():
        scene = BF.beamform(dataRC, RP.pixels_3D_bf, group_delay=GD, temps=temps)
        scene = scene.detach().cpu().numpy()

        np.save(os.path.join(save_directory, 'scene' + '.npy'), scene)

        save_sas_plot(c2g(np.absolute(scene), RP.circle_indeces, SIZE_W, SIZE_H),
                      os.path.join(save_directory, 'scene_abs' + '.png'))

        scene = c2g(scene, RP.circle_indeces, SIZE_W, SIZE_H)

    ############ COMPUTE THE PSF #############################
    print("Creating PSF... This will take a while")
    wfms, psf = compute_psf(scene_dir, device, SIZE_W, SIZE_H)
    save_sas_plot(np.abs(psf), os.path.join(save_directory, 'psf_abs' + '.png'))
    np.save(os.path.join(save_directory, 'psf.npy'), psf)

    ############### Deconvolve the scene #####################

    to_be_deconvolved = [{
        'scene': scene,
        'psf': psf,
        'gt': None
    }]

    DM = DeconvMethods(deconv_config=deconv_config,
                       to_be_deconvolved=to_be_deconvolved,
                       deconv_dir=save_directory,
                       circular=True,
                       device='cuda:0')

    DM.run_all_methods()







