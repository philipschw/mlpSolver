"""
The main file to run MLP solver to solve semilinear parabolic partial differential equations (PDEs)
with gradient-independent as well as gradient-dependent nonlinearity.
"""

import json
import munch
import os
import logging
import warnings
import time

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np

import equation as eqn
from solver import MLPSolver

# try to load module which requires sdeint2 (sdeint extension)
samplingAvailable = True
try:
    from generateSamples import start_mlp_generateSamples
except ModuleNotFoundError:
    samplingAvailable = False
    

# define flags
flags.DEFINE_string('config_path', 'configs/semilinear_blackscholes.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
flags.DEFINE_string('sample_path', 'test',
                    """The path to load sample data.""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array

#turn off warnings on Windows 10
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 


def main(argv):
    del argv
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, time.strftime("%Y%m%d%H%M%S.{}".format(repr(time.time()).split('.')[1][:3]), time.gmtime()) + "_" + FLAGS.exp_name)

    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
        for name in dir(config) if not name.startswith('__')),outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    for d in config.eqn_config.dim:
        for n in config.eqn_config.num_iteration:
            for num_gridpoint in config.eqn_config.num_gridpoint:
                samples = None
                # check if sampling is available (i.e. if sdeint2 is imported)
                if(samplingAvailable):
                    # generate samples based on fixed d, n, and number gridpoints
                    if(FLAGS.sample_path != 'test'):
                            # if we load data from disk, then only one realization is allowed
                            if(config.eval_config.num_realization != 1 or len(config.eqn_config.num_gridpoint) != 1):
                                raise ValueError("For loaded data, only one realization (and length of num_gridpoint is 1) is allowed. Please modify your config file.")
                                exit()
                            else:
                                samples = {}
                                samples[1] = np.load(FLAGS.sample_path, allow_pickle=True).item()
                    
                    elif(True in config.eval_config.advanced):
                        samples = {}
                        logging.info('Pre-Generating Samples for advanced mode.')
                        for theta in range(1,config.eval_config.num_realization + 1):
                            samples[theta] = (start_mlp_generateSamples(total_time=config.eqn_config.total_time,
                                                                        start_time=config.eqn_config.start_time,
                                                                        M=n,
                                                                        n=n,
                                                                        num_gridpoint=num_gridpoint,
                                                                        dim=d,
                                                                        active=config.eval_config.advanced,
                                                                        time_dist_exponent=config.eqn_config.time_dist_exponent))
                            if(config.eval_config.saveSamples):
                                path_prefix_sample = (os.path.join(FLAGS.log_dir, time.strftime("%Y%m%d%H%M%S.{}".format(repr(time.time()).split('.')[1][:3]), time.gmtime())
                                                                                                                    + "_" + FLAGS.exp_name
                                                                                                                    + "_d" + str(d)
                                                                                                                    + "_n" + str(n)
                                                                                                                    + "_grid" + str(num_gridpoint)
                                                                                                                    + "_realization" + str(theta)))
                                np.save('{}_sample_history.npy'.format(path_prefix_sample), samples[theta])
                else:
                    message="""WARNING sdeint2 package could not be loaded. Pre-Generation of samples for comparison not avaible. Undefined behavior for usage of methods NOT from sdeint. Only limited functionality is available for mlpSolver."""
                    print(message)
                # sample different methods
                for method in config.eqn_config.samplingMethod:
                    path_prefix_realization = (os.path.join(FLAGS.log_dir, time.strftime("%Y%m%d%H%M%S.{}".format(repr(time.time()).split('.')[1][:3]), time.gmtime())
                                                                                                        + "_" + FLAGS.exp_name
                                                                                                        + "_d" + str(d)
                                                                                                        + "_n" + str(n)
                                                                                                        + "_method" + method 
                                                                                                        + "_grid" + str(num_gridpoint)))
                    logging.info('Begin to solve %s with dimension %d, iterations n=%d, method=%s, and no.gridpoints=%d' % (config.eqn_config.eqn_name, d, n, method, num_gridpoint))
                    
                    mlp = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config, d, n, method, num_gridpoint)
                    mlp_solver = MLPSolver(config, mlp)
                    training_history = mlp_solver.train(samples=samples)
            
                    np.savetxt('{}_training_history.csv'.format(path_prefix_realization),
                               np.vstack(training_history[0:-1]),
                               fmt=['%d', '%s', '%f', '%s'],
                               delimiter=",",
                               header='realization,sol,elapsed_time,cost',
                               comments='')
                    
                    header = ''
                    if not os.path.isfile('{}_evaluation.csv'.format(path_prefix)): 
                        header = 'dimension,n,average_solution,reference_solution,L1error,rel1error,L2error,rel2error,empiricalSD,elapsed_time'
                    with open('{}_evaluation.csv'.format(path_prefix), 'ab') as file:               
                        np.savetxt(file,
                                   np.array(training_history[-1]).reshape((1,10)),
                                   fmt=['%d', '%d','%s', '%s', '%f', '%f', '%f', '%f', '%f', '%f'],
                                   delimiter=",",
                                   header=header,
                                   comments='')
                    logging.info('\n')
                    
if __name__ == '__main__':
    app.run(main)
