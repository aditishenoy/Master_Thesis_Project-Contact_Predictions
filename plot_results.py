import glob

#import pylab as plt
from matplotlib import pyplot as plt
import numpy as np

p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_all_Mplus_AltRegDouble12_mae_trained')
p_2= glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_all_model12_mae_trained')
p_3= glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_all_M07_R05_D01_E50_mae_trained')
p_4= glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/results_wednesday24/results_8_all_M12_R05_D01_Test_Epochs_mae_trained')
#p_5= glob.glob('


for f in (p_1+p_2):
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', ls='-', alpha=1)
    fig1.suptitle('Precison vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Precison')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 2])
    fig1.suptitle('Recall vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Recall')

plt.show()

'''
for f in p_2:
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', ls='-', alpha=1)
    fig1.suptitle('Precison vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Precison')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 2])
    fig1.suptitle('Recall vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Recall')

plt.show()

for f in p_3:
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', ls='-', alpha=1)
    fig1.suptitle('Precison vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Precison')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 2])
    fig1.suptitle('Recall vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Recall')

plt.show()
for f in p_4:
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', ls='-', alpha=1)
    fig1.suptitle('Precison vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Precison')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 2])
    fig1.suptitle('Recall vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Recall')

plt.show()
'''
