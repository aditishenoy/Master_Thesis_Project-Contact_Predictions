# Script for plotting the precision-recall curve and calcualting the AUC

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

import glob
import pylab as plt
import numpy as np

p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_roc_monday29/regroc_prec_rec_values')

for f in (p_1):
    d = np.loadtxt(f)
    
    fig1 = plt.figure()
    plt.plot([0, 1], [0.1, 0.1], linestyle='--')
    precision_u = d[:, 0]
    recall_u = d[:, 1]
    precision_up = d[:, 2]
    recall_up = d[:, 3]
    plt.plot(d[:, 1], d[:, 0], color='b', marker='.', label='U-Net')
    plt.plot(d[:, 3], d[:, 2], color='g', marker='.', label='U-Net++')
    auc_u = auc(recall_u, precision_u)
    auc_up = auc(recall_up, precision_up)
    print (auc_u)
    print ()
    print (auc_up)

    fig1.suptitle('Precision-Recall curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/Prec-Rec Curve (Reg Models)')

plt.show()


