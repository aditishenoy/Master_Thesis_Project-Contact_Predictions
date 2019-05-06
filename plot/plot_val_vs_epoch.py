# Plotting Validation Loss vs Epoch; Learning Rate vs Epoch; Precision vs Epoch; Recall vs Epoch
import glob

import pylab as plt
import numpy as np


p_1 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/reg_loss_lr_values')
p_2 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/reg_prec_rec_values')
p_3 = glob.glob('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/all_val_loss_values')

# Validation Loss vs Epoch (U-net & U-net++)
# Learning Rate vs Epoch (U-net & U-net++)

for f in (p_1):
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', label='U-Net')
    plt.plot(d[:, 0], d[:, 2], color='g', label='U-Net++')
    fig1.suptitle('Validation loss vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation loss')
    plt.legend(loc='upper left')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/Val_loss_Epochs_Reg')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 3], color='b', label='U-Net')
    plt.plot(d[:, 0], d[:, 4], color='g', label='U-Net++')
    fig1.suptitle('Learning rate vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Learning rate')
    plt.legend(loc='lower left')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/LearningRate_Epochs_Reg')

plt.show()


# Precision vs Epoch (U-net & U-net++)
# Recall vs Epoch (U-net & U-net++)

for f in p_2:
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', label='U-Net')
    plt.plot(d[:, 0], d[:, 3], color='g', label='U-Net++')
    fig1.suptitle('Precison vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Precison')
    plt.legend(loc='upper left')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/Precision_Epochs_Reg')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 2], color='b', label='U-Net')
    plt.plot(d[:, 0], d[:, 4], color='g', label='U-Net++')
    fig1.suptitle('Recall vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Recall')
    plt.legend(loc='lower right')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/Recall_Epochs_Reg')

plt.show()

# Validation Loss vs Epoch (all models)
# Learning Rate vs Epoch (all models)

for f in (p_3):
    d = np.loadtxt(f)

    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 1], color='b', label='Reg Model (U-Net)')
    plt.plot(d[:, 0], d[:, 2], color='g', label='Clas Model (12 bins)')
    plt.plot(d[:, 0], d[:, 3], color='r', label='Clas Model (26 bins)')
    plt.plot(d[:, 0], d[:, 4], color='c', label='Clas Model (7 bins)')
    plt.plot(d[:, 0], d[:, 5], color='m', label='Reg Model (U-Net++)')
    fig1.suptitle('Validation loss vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation loss')
    plt.legend(loc='lower right')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/Val_loss_allModels_Epochs_Reg')


    fig1 = plt.figure()
    plt.plot(d[:, 0], d[:, 6], color='b', label='Reg Model (U-Net)')
    plt.plot(d[:, 0], d[:, 7], color='g', label='Clas Model (12 bins)')
    plt.plot(d[:, 0], d[:, 8], color='r', label='Clas Model (26 bins)')
    plt.plot(d[:, 0], d[:, 9], color='c', label='Clas Model (7 bins)')
    plt.plot(d[:, 0], d[:, 10], color='m', label='Reg Model (U-Net++)')
    fig1.suptitle('Learning rate vs Epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Learning rate')
    plt.legend(loc='lower right')
    plt.savefig('/home/ashenoy/ashenoy/Thesis_unetplus_V.0.1/images/LearningRate_allModels_Epochs_Reg')

plt.show()

