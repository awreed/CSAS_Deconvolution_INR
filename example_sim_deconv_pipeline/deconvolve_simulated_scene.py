import sys
import numpy as np
sys.path.append('../')
from sim_csas_package import SimulateMeasurements, CreatePSF
from deconv_methods.deconv_methods import DeconvMethods

if __name__ == '__main__':
    sim_config = 'config/simulation.ini'
    sys_config = 'config/system_parameters.ini'
    deconv_config = 'config/deconv.ini'
    save_img_dir = 'sim_images'
    save_data_dir = 'sim_data'
    deconv_dir = 'deconv_dir'

    # Create dataset(sim_config, sys_config, save_img_dir, save_data_dir) -> returns nothing
        # Simulate Measurements(sys_config, img) -> raw_wfms, mf_wfms
        # BF Measurements(sys_config)


    print("Simulating Scenes")
    # Simulate measurements using the specifications given in the sim_config file
    SM = SimulateMeasurements(sim_config=sim_config, sys_config=sys_config,
                              save_img_dir=save_img_dir, save_data_dir=save_data_dir)
    SM.run()

    #psf_wfms, _ = sim_measurments.get_waveforms_from_image(image)

    #BF = Beamformer(sys_config)
    #BF.beamform(psf_wfms)

    print("Creating PSF")
    PSF = CreatePSF(sys_config=sys_config, save_img_dir=save_img_dir, save_data_dir=save_data_dir)
    PSF.run()

    to_be_deconvolved = [{'scene': np.load('sim_data/beamformed_scatterers_0.npy'),
                          'psf': np.load('sim_data/psf.npy'),
                          'gt': np.load('sim_data/gt0.npy')},

                         {'scene': np.load('sim_data/beamformed_scatterers_1.npy'),
                          'psf': np.load('sim_data/psf.npy'),
                          'gt': np.load('sim_data/gt1.npy')}]

    # Pass in the scene and the psf for the scene.
    DM = DeconvMethods(deconv_config=deconv_config,
                       sys_config=sys_config,
                       to_be_deconvolved=to_be_deconvolved,
                       deconv_dir=deconv_dir)

    DM.run_all_methods()
