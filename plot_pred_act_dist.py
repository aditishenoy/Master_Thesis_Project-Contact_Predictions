import glob

#import pylab as plt
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interpn

#p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/pred_act_residues_values_M07_R05_D01_E50_mae_trained')
#p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/pred_act_residues_values_M12_R05_D01_Test_Epochs_mae_trained')
p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/pred_act_residues_values_M26_R05_D01_E50_mae_trained')


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    return ax


for f in (p_1):
    d = np.loadtxt(f)
    x = d[:,0]
    y = d[:,1]

density_scatter( x, y, bins = [30,30] )
plt.suptitle('Predicted Distance Vs Actual Distance')
plt.xlabel('Actual Distance')
plt.ylabel('Predicted Distance')
plt.savefig('images/Pred_Act_Distance_26')
plt.show()


'''
plt.scatter(x,y)
fig1.suptitle('Predicted Distance Vs Actual Distance')
plt.xlabel('Actual Distance')
plt.ylabel('Predicted Distance')
plt.savefig('images/Pred_Act_Distance_07')
#plt.savefig('images/Pred_Act_Distance_12')
#plt.savefig('images/Pred_Act_Distance_26')
plt.show()

from ggplot import *

'''
