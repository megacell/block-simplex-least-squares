from scipy import io as sio
import python.util as util
import python.config as config
import numpy as np
from matplotlib import pyplot

def allLinksB():
    dir = '{0}/{1}/{2}/{3}/experiment2_all_link_matrices_routes_2000.mat'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR,950)
    matrix = sio.loadmat(dir)['b']
    return matrix

def allLinksA(density, routes):
    dir1 = '{0}/{1}/{2}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR)
    dir2 = '{0}/experiment2_all_link_matrices_routes_{1}.mat'.format(density,routes)
    matrix = sio.loadmat('{0}/{1}'.format(dir1, dir2))['A']
    return matrix

def readZ(density, routes):
    dir1 = '{0}/{1}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR)
    dir2 = '{0}/output_waypoints{1}.mat'.format(density,routes)
    matrix = sio.loadmat('{0}/{1}'.format(dir1, dir2))['x']
    return matrix

def readB(density, routes):
    dir1 = '{0}/{1}/{2}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR, config.ALL_LINK_DIR)
    dir2 = '{0}/experiment2_all_link_matrices_routes_{1}.mat'.format(density,routes)
    matrix = sio.loadmat('{0}/{1}'.format(dir1, dir2))['b']
    return matrix

def readX(density,routes):
    dir1 = '{0}/{1}'.format(config.PLOT_DIR, config.EXPERIMENT_MATRICES_DIR)
    dir2 = '{0}/experiment2_waypoints_matrices_routes_{1}.mat'.format(density,routes)
    A, b, N, block_sizes, x_true, nz, f = util.load_data('{0}/{1}'.format(dir1, dir2))
    x0 = np.array([util.block_e(block_sizes - 1, block_sizes)])
    return x0.T + N*readZ(density,routes).T

def GEH(b_estimate, b_true):
    diff = b_estimate - b_true
    plus = b_estimate + b_true
    GEH = np.sqrt(2 * diff**2 / (np.maximum(np.zeros(plus.shape),plus) + 1e-12))
    return GEH

def bin(f, b_estimate, b_true):
    bin_b_estimate = [b for b, x in zip(b_estimate,b_true) if f(x)]
    bin_b_true = [b for b in b_true if f(b)]
    return bin_b_estimate, bin_b_true

def percent_above_5(geh):
    return len(list(filter(lambda x:x>5, geh)))/float(len(geh))

def GEH_bin2700(b_est, b_true):
    b_est, b_true = bin(lambda x: x > 2700, b_est, b_true)
    print len(b_est)
    return GEH(np.array(b_est), np.array(b_true))

def GEH_bin700_2700(b_est, b_true):
    b_est, b_true = bin(lambda x:700 < x < 2700, b_est, b_true)
    print len(b_est)
    return GEH(np.array(b_est), np.array(b_true))

def GEH_bin700(b_est, b_true):
    b_est, b_true = bin(lambda x:700 > x, b_est, b_true)
    print len(b_est)
    return GEH(np.array(b_est), np.array(b_true))

def percent_under(n, xs):
    return sum(1.0 for x in xs if x < n)/len(xs)

def percent_under5(xs):
    return sum(1.0 for x in xs if x < 5)/len(xs)

def geh_by_bin(b_est, b_true):
    geh = percent_under(4,GEH_bin2700(b_est, b_true))
    geh2 = percent_under(4,GEH_bin700_2700(b_est, b_true))
    geh3 = percent_under(4,GEH_bin700(b_est, b_true))
    print geh, geh2, geh3
    return geh, geh2, geh3

def plot_waypoint_vs_geh(num_routes = 50):
    geh_values = []
    waypoints = config.WAYPOINT_DENSITIES[0:len(config.WAYPOINT_DENSITIES) - 1]
    for d in waypoints:
        b_est = allLinksA(d,num_routes) * readX(d,num_routes)
        b_true = allLinksB().T
        b_t2 = readB(d, num_routes).T
        geh_values.append(geh_by_bin(b_est, b_true))

    pyplot.plot([0,4000],[.85,.85],'--k')
    geh_values =np.array(geh_values)
    p = pyplot.plot(waypoints, geh_values,'-o')
    pyplot.legend(p,['>2700vph','2700-700vph','<700vph'], fontsize=22)
    pyplot.ylim([0,1.25])
    pyplot.xlim([0,4000])

    pyplot.title('MATSim link flow error', fontsize=22, weight='bold')
    pyplot.xlabel('Number of cells', fontsize=22)
    pyplot.ylabel('% (GEH < 5)', fontsize=22)

    pyplot.show()

def plot_routes_vs_geh(waypoint_density):
    routes = [3,10,20,30,40,50]
    geh_values = []
    geh_unmodeled = []
    for r in routes:
        b_est = allLinksA(waypoint_density,r) * readX(waypoint_density,r)
        b_true = allLinksB().T
        b_t2 = readB(waypoint_density, r).T
        geh_values.append(geh_by_bin(b_est, b_true))
        geh_unmodeled.append(geh_by_bin(b_est, b_t2))
    pyplot.plot([0,60],[.85,.85],'--k',linewidth=2)
    geh_values =np.array(geh_values)
    geh_unmodeled=np.array(geh_unmodeled)
    sio.savemat(config.PLOT_DIR+'/gehdata.mat',
                {'waypoint_density':waypoint_density, 'routes':routes,'per_geh_model_error':geh_values,'per_geh_no_model_error':geh_unmodeled})
    p = pyplot.plot(routes, geh_values,'-o', linewidth=2)
    pyplot.legend(p,['>2700vph','700-2700vph','<700vph'], fontsize=22, loc=4)
    colors=['b','g','r']
    for i,c in zip(range(3),colors):
        pyplot.plot(routes, geh_unmodeled[:,i],'--o'+c, linewidth=2)
    pyplot.ylim([0,1.1])
    pyplot.xlim([0,60])

    pyplot.title('MATSim link flow error', fontsize=22, weight='bold')
    pyplot.xlabel('Routes', fontsize=22)
    pyplot.ylabel('% (GEH < 4)', fontsize=22)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    pyplot.show()

plot_routes_vs_geh(950)