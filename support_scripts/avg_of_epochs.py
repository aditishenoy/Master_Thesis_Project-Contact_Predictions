# Support script for calculating the average of each of the individual metrics 
import glob
import numpy as np 

p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_all_M26_R05_D01_E50_mae_trained')
p_2 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_short_M26_R05_D01_E50_mae_trained')
p_3 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_medium_M26_R05_D01_E50_mae_trained')
p_4 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_long_M26_R05_D01_E50_mae_trained')

avg_prec = []
avg_rec = []
avg_f1 = []
avg_ab_err = []
avg_rel_err = []

def mean(lst): 
    return sum(lst) / len(lst) 

for f in (p_4):
    d = np.loadtxt(f)
    avg_prec = mean(d[:,1])
    avg_rec = mean(d[:,2])
    avg_f1 = mean(d[:,3])
    avg_ab_err = mean(d[:,6])
    avg_rel_err = mean(d[:,7])

print (avg_prec)
print (avg_rec)
print (avg_f1)
print (avg_ab_err)
print (avg_rel_err)
