"-------------------------------------------------------------------"
## Plot Scatter Plot for Predicted Distance vs Actual Distance 

import glob
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interpn
import math

import numpy as np
import itertools

p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/reg_pred_act_ALLall_values_model12_mae_trained')

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( np.array(x[1:100000000]),np.array(y[1:100000000]), bins = bins)
    z = interpn( ( 0.05*(x_e[1:] + x_e[:-1]) , 0.05*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    return ax


for f in (p_1):
    d = np.loadtxt(f)
    x = d[:,0]
    for i in x:
        i = abs(i)
    y = d[:,1]
    for i in y:
        i = abs(i)

density_scatter( x, y, bins = [30,30] )
plt.suptitle('Predicted Distance Vs Actual Distance for all contacts (U-Net)')
plt.xlabel('Actual Distance')
plt.ylabel('Predicted Distance')
plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/lat_images/ALL/model_ALLall')
#plt.show()

'''
"-------------------------------------------------------------------"

## Plot bar plot for probability distribution for all the bins

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

bins = [2, 5, 7, 9, 11, 13, 15]
#bins = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
#bins = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]

y_pos = np.arange(len(bins))
## MODEL 07
#Incorrectly predicted (Actual Value = 4.775731, Predicted Value = 13.201516206609085)
#performance = [0.00121875, 0.02300803, 0.04109265, 0.08310042, 0.1170807, 0.12844664, 0.6060528]
#Correctly predicted contact (Actual value = 7.013706, Predicted value = 7.825837418437004)
performance = [0.02045989, 0.33463833, 0.28087327, 0.1577469,  0.07977389, 0.02461922, 0.10188856] 

## MODEL 12
#Incorrectly predicted (Actual Value = 4.4413605, Predicted Value =  15.039671569364145)
#performance = [0.00255118, 0.00309624, 0.00356882, 0.0046248,  0.00690324, 0.00989708, 0.01215325, 0.01296933, 0.01501255, 0.0196325, 0.02243174, 0.8871593]
#Correctly predicted contact (Actual value = 7.181674, Predicted value =7.295131782157114)
#performance = [1.6374025e-03, 1.4637760e-02, 3.1689158e-01, 6.3047445e-01, 1.5119189e-02, 1.9384867e-03, 2.6481724e-03, 1.8701594e-03, 5.0919340e-04, 6.3119660e-04, 4.0849810e-04, 1.3233954e-02]

## MODEL 26
#Incorrectly predicted (Actual Value = 7.6430902, Predicted Value = 2.0002495030995773)
#performance = [9.9998033e-01, 3.7806000e-08, 9.2701015e-08, 1.9704113e-07, 2.2723439e-07,2.0417041e-07, 8.5642085e-08, 1.5224155e-07, 1.7324051e-07, 1.9693950e-07, 1.8356168e-07, 2.7882871e-07, 3.2809371e-07, 3.4138870e-07, 4.7730663e-07, 4.7727701e-07, 3.4675256e-07, 2.5135193e-07, 2.8311834e-07, 2.9609492e-07, 5.2018936e-07, 2.8421451e-07, 2.5025170e-07, 2.0290096e-07, 1.7283635e-07, 1.3735030e-05]
#Correctly predicted contact (Actual value = 7.012482, Predicted value = 8.02883640100481)
#performance = [0.00047604, 0.00428063, 0.00429537, 0.00549353, 0.01148515, 0.04064604, 0.1326905,  0.28674737, 0.27311608, 0.1141442,  0.02606971, 0.00673238, 0.00527186, 0.00628595, 0.00479341, 0.00545011, 0.00378869, 0.00501036, 0.00482416, 0.00352946, 0.00296577, 0.00212278, 0.00219996, 0.00153883, 0.00179232, 0.04424928]


#fig, ax = plt.subplots()
#fig.autofmt_xdate()

plt.bar(y_pos, performance,bottom=None, align='center', alpha=0.5)
plt.xticks(y_pos, bins)
plt.xlabel('Midpoint of each bin')
plt.ylabel('Probability')
plt.suptitle('Probability distribution for a correctly predicted contact (Model07)')
plt.savefig('images/Correct_pred_contact_07') 
plt.show()

'''

"-------------------------------------------------------------------"

