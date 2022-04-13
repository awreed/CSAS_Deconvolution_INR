import sys
sys.path.append('../')
from sim_csas_package import SimulateMeasurements, CreatePSF

if __name__ == '__main__':
    sim_config = 'config/simulation.ini'
    sys_config = 'config/system_parameters.ini'
    save_img_dir = 'sim_images'
    save_data_dir = 'sim_data'

    print("Simulating Scenes")
    # Simulate measurements using the specifications given in the sim_config file
    SM = SimulateMeasurements(sim_config=sim_config, sys_config=sys_config,
                              save_img_dir=save_img_dir, save_data_dir=save_data_dir)
    SM.run()

    print("Creating PSF")
    PSF = CreatePSF(sys_config=sys_config, save_img_dir=save_img_dir, save_data_dir=save_data_dir)
    PSF.run()



