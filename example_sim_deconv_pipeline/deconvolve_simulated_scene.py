import sys
sys.path.append('../')
from sim_csas_package import SimulateMeasurements

if __name__ == '__main__':
    sim_config = 'sim_config.ini'

    # Simulate measurements using the specifications given in the sim_config file
    SM = SimulateMeasurements(sim_config)
    SM.run()



