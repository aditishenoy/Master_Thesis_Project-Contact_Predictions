# Support script for calculating the average of each of the individual metrics 
import glob
import numpy as np 
import math

p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/clas_pred_act_values_M26_R05_D01_E50_mae_trained')
#p_2 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_short_M26_R05_D01_E50_mae_trained')
#p_3 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_medium_M26_R05_D01_E50_mae_trained')
#p_4 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_long_M26_R05_D01_E50_mae_trained')


avg_rmsd = []
avg_mae = []

def mean(lst): 
    return sum(lst) / len(lst) 

for f in (p_1):
    d = np.loadtxt(f)
    act = d[:,0]
    pred = d[:,1]
    avg_rmsd = mean((abs(act - pred))*(abs(act - pred)))
    avg_mae = mean(abs(act - pred))

print (math.sqrt(avg_rmsd))
print (avg_mae)
