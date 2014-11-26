ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']
DATA_DIR = '/home/lei/traffic/datasets/Phi'
EXPERIMENT_MATRICES_DIR = 'hierarchy/experiment_matrices'
ALL_LINK_DIR = 'AllLinks'
ESTIMATION_INFO_DIR = 'estimation_info'
PLOT_DIR = DATA_DIR + '/plots'
WAYPOINT_DENSITIES = [3800,3325,2850,2375,1900,1425,950,713,475,238]
import os
# The directory must exist for other parts of this application to function properly
assert(os.path.isdir(DATA_DIR))
assert(os.path.isdir(DATA_DIR+'/'+EXPERIMENT_MATRICES_DIR))
