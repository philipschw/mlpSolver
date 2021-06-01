"""
The main file to run MLP solver to solve semilinear parabolic partial differential equations (PDEs)
with gradient-independent nonlinearity.

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


flags.DEFINE_string('config_path', 'configs/semilinear_blackscholes.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
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
	path_prefix = os.path.join(FLAGS.log_dir, time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_" + FLAGS.exp_name)

	with open('{}_config.json'.format(path_prefix), 'w') as outfile:
		json.dump(dict((name, getattr(config, name))
		for name in dir(config) if not name.startswith('__')),outfile, indent=2)

	absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
	absl_logging.set_verbosity('info')

	for d in config.eqn_config.dim:
		for n in config.eqn_config.num_iteration:
			for method in config.eqn_config.samplingMethod:
				runNextNot = False
				for num_gridpoint in config.eqn_config.num_gridpoint:
					it = num_gridpoint
					if runNextNot:
						break
					if method == "explicit":
						runNextNot = True
						it = 1
						
					path_prefix_realization = os.path.join(FLAGS.log_dir, time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_" + FLAGS.exp_name + "_d" + str(d) + "_n" + str(n) + "_method" + method + "_grid" + str(it))
					logging.info('Begin to solve %s with dimension %d, iterations n=%d, method=%s, and no.gridpoints=%d' % (config.eqn_config.eqn_name, d, n, method, it))
					mlp = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config, d, n, method, it)
					mlp_solver = MLPSolver(config, mlp)
					training_history = mlp_solver.train()
			
					np.savetxt('{}_training_history.csv'.format(path_prefix_realization),
							   np.vstack(training_history[0:-1]),
							   fmt=['%d', '%s', '%f'],
							   delimiter=",",
							   header='realization,sol,elapsed_time',
							   comments='')
					
					header = ''
					if not os.path.isfile('{}_evaluation.csv'.format(path_prefix)):	
						header = 'dimension,n,average_solution,reference_solution,rel2error,numberRVrealizations,elapsed_time'
					with open('{}_evaluation.csv'.format(path_prefix), 'ab') as file:				
						np.savetxt(file,
								   np.array(training_history[-1]).reshape((1,7)),
								   fmt=['%d', '%d','%s', '%s', '%f', '%d', '%f'],
								   delimiter=",",
								   header=header,
								   comments='')
					logging.info('\n')
					
if __name__ == '__main__':
	app.run(main)
