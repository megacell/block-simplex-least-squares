__author__ = 'lei'
import python.config as config
import pickle
import scipy.io as sio
import numpy as np

from matplotlib import pyplot
def correct_for_unmodeled_flow(p, route_number):
    modeled_flow = sio.loadmat(config.PLOT_DIR+'/modeled_flow.mat')
    total_flow = modeled_flow['total_flow']
    captured_flow = modeled_flow['captured_flow']
    flow_in_routes = np.squeeze(captured_flow)[route_number - 1]
    unmodeled_flow = total_flow - flow_in_routes
    return ((p*flow_in_routes + unmodeled_flow)/total_flow).flat[0]

def convert_to_waypoint_vs_percent_error(stats, route_index):
    keys = sorted(stats.keys())
    values = [stats[d][route_index]['flow_per_error'] for d in keys]
    return keys, values

def plot_waypoint_density_vs_error(plot,correction = lambda x, y: x):
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['m', 'c', 'b', 'k', 'g', 'r']
    indexes = list(range(6))
    routes = [3,10,20,30,40,50]
    plots = []
    output_mat = dict()
    for c, i in zip(colors, indexes):
        x, y = convert_to_waypoint_vs_percent_error(stats, i)
        output_mat['y'+str(routes[i])] = y
        y = [correction(y_i, routes[i]) for y_i in y]
        output_mat['waypoints'] = x
        output_mat['y_corr'+str(routes[i])] = y
        p, = plot(x[1:], y[1:], '-o' + c, linewidth=2)
        plot([0,4000], y[0]*np.ones((2,1)), '--' + c,linewidth=2)
        plots.append(p)
    sio.savemat(config.PLOT_DIR + '/waypoint_plot.mat', output_mat)
    pyplot.legend(plots, ['{0} Routes'.format(i) for i in routes],fontsize=22)
    pyplot.xlabel('Cells',fontsize=22)
    pyplot.ylabel('Relative Error',fontsize=22)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    pyplot.ylim([0,5])
    pyplot.xlim([0,4000])


def plot_waypoint_density_vs_error2(plot,correction = lambda x, y: x):
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['m', 'c', 'b', 'k', 'g', 'r']
    indexes = list(range(6))
    routes = [3,10,20,30,40,50]
    plots = []
    for c, i in zip(colors, indexes):
        x, y = read_data(config.WAYPOINT_DENSITIES[i])
        y = [correction(y_i, routes[i]) for y_i in y]
        p, = plot(x, y, '-o' + c, linewidth=2)
        plots.append(p)

    pyplot.legend(plots, ['{0} Routes'.format(i) for i in routes],fontsize=22)
    pyplot.xlabel('Cells',fontsize=22)
    pyplot.ylabel('Relative Error',fontsize=22)
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    pyplot.ylim([0,5])
    pyplot.xlim([0,4000])


def convert_to_route_vs_percent_error(stats, waypoint):
    routes = [3,10,20,30,40,50]
    return routes, [s['flow_per_error'] for s in stats[waypoint]]
def read_data(offset):
    routes = [3,10,20,30,40,50]
    blarg = [3800,3325,2850,2375,1900,1425,950,713,475,238]
    offset = blarg.index(offset)
    f = open('/home/lei/errordata.txt')
    s = f.read().split('\n')
    print s
    result = []
    for i in range(len(blarg)):
        result.append(float(s[6*i+offset]))
    return blarg, result

def plot_routes_vs_error(plot, correction = lambda x, y: x):
    f = open(config.PLOT_DIR + '/stats.pkl')
    stats = pickle.load(f)
    colors = ['r', 'g', 'b','c','m','k','r','g','b']
    routes = [3,10,20,30,40,50]
    waypoints = config.WAYPOINT_DENSITIES
    plots = []
    for c, w in zip(colors, waypoints):
        x, y = read_data(w) #convert_to_route_vs_percent_error(stats, w)

        y = [correction(y_i, routes[i]) for i,y_i in enumerate(y)]
        p, = plot(x, y, '-o' + c, weight='2pt')
        plots.append(p)
    pyplot.legend(plots, ['Cell Density-{0}'.format(i) for i in waypoints])
    pyplot.xlabel('Routes')
    pyplot.ylabel('Route Flow Percent Error')
    pyplot.ylim([0,5])

plot_waypoint_density_vs_error(pyplot.semilogy)
pyplot.title('MATSim route flow error from cell + OD data \n(AM)',fontsize=22, weight='bold')
pyplot.show()
pyplot.close('all')


#plot_waypoint_density_vs_error(pyplot.semilogy,correct_for_unmodeled_flow)
#pyplot.title('MATSim route flow error from cell + OD data \n(AM, model error)',fontsize=22, weight='bold')
#pyplot.show()
pyplot.close('all')
