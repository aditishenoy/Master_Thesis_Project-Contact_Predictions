'-------------------------------------------------------------------'
import os
import glob
import functools
import sys
from keras.models import load_model
import numpy as np
from Bio import pairwise2

import pylab as plt

import tqdm

#Number of bins used for classification (Based on model_name)
n_bins = int(sys.argv[1])

#Distance threshold to calculate all the other measures (8 or 15)
#thres = int(sys.argv[2])
thres = 8

#The value n which is multiplied with L (Length of protein) to get the top n*L contacts
#threshold_length = float(sys.argv[3])
threshold_length = 1

#What type of protein length is it (short, medium, long, all)
#range_mode = sys.argv[2]
range_mode = 'all'


if n_bins == 1:
    model_name = 'model12_mae_trained'
    model_n = 'model12'
    selected_epoch = 14


elif n_bins == 2:
    model_name = 'Mplus_AltRegDouble12_mae_trained'
    model_n = 'Mplus_AltRegDouble12'
    selected_epoch = 75


elif n_bins == 7:
    model_name = 'M07_R05_D01_E50_mae_trained'
    model_n = 'M07_R05_D01_E50'
    bins = [2, 5, 7, 9, 11, 13, 15]
    prob_len = 3
    selected_epoch = 25

elif n_bins == 26:
    model_name = 'M26_R05_D01_E50_mae_trained'
    model_n = 'M26_R05_D01_E50'
    bins = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]
    prob_len = 9
    selected_epoch = 32

elif n_bins == 12:
    model_name = 'M12_R05_D01_Test_Epochs_mae_trained'
    model_n = 'M12_R05_D01_Test_Epochs'
    bins = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
    prob_len = 4
    selected_epoch = 25

else:

    raise ValueError('Invalid number of bins (n_bins): {}'.format(n_bins))


assert 0 < threshold_length < 4., 'Invalid threshold_length to contact top contacts'

assert range_mode in ('short', 'medium', 'long', 'all'), range_mode

m = load_model('{}.h5'.format(model_name))

if (n_bins ==1) or (n_bins == 2):
    weights_path = 'regression/models/{}/{}_epo{:02d}-*.h5'.format(model_name, model_n, selected_epoch)
else:
    weights_path = 'classification/models/{}/{}_epo{:02d}-*.h5'.format(model_name, model_n, selected_epoch)

weights = glob.glob(weights_path)[0]
m.load_weights(weights)


'-------------------------------------------------------------------'

def _strip(x):
    return len(x[1].strip('-'))

three_to_one = {'ASP': 'D', 'GLU': 'E', 'ASN': 'N', 'GLN': 'Q', 'ARG': 'R', 'LYS': 'K', 'PRO': 'P', 'GLY': 'G',
                'CYS': 'C', 'THR': 'T', 'SER': 'S', 'MET': 'M', 'TRP': 'W', 'PHE': 'F', 'TYR': 'Y', 'HIS': 'H',
                'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'MSE': 'M'}

def pick_best_alignment(align, sequence):
    # Pick the best alignment mode:
    align_1 = [al for al in align if al[0] == sequence]
    if align_1:
        if len(align_1) == 1:
            # There is only one option
            al = align_1[0]
        else:
            # Pick the one that has the gaps on the pdb at the tails
            al = min(align_1, key=_strip)
    else:

        # warnings.warn(RuntimeWarning(cif_file))
        al = align[0]
    return al

@functools.lru_cache(maxsize=512)
def parse_pdb(pdb):
    ca = dict()
    cb = dict()

    this_chain = pdb.split('/')[-2][-1]
    fa = pdb.replace('native.pdb', 'sequence.fa')
    seq = ''.join(line.strip() for line in open(fa) if not line.startswith('>'))
    pdb_seq = dict()
    pdb_numbers = []
    for line in open(pdb):
        if line.startswith('ATOM'):
            type_ = line[13:15]
            chain = line[21:22]
            if type_ == 'CA' and chain == this_chain:
                resi = int(line[22:26])
                if resi not in pdb_seq:
                    pdb_seq[resi] = three_to_one[line[17:20]]
                    pdb_numbers.append(resi)

    pdb_seq_merged = ''.join(pdb_seq[k] for k in sorted(pdb_seq))

    aligned = pairwise2.align.globalmd(seq, pdb_seq_merged, 5, -1, -5, -3, -0.5, -0.1)
    if len(aligned) == 0:
        aligned = pairwise2.align.globalmd(seq, pdb_seq_merged, 5, -1, -3, -3, -0.5, -0.1)
        if len(aligned) == 0:
            aligned = pairwise2.align.localmd(seq, pdb_seq_merged, 10, -1, -3, -3, -0.5, -0.1)

    aligned = pick_best_alignment(aligned, seq)

    res_mapping = dict()
    j = 0
    for i, aa in enumerate(aligned[1]):
        if aa != '-':
            skipped = aligned[0][:j].count('-')  # Count for gaps in the sequence
            pdb_number = pdb_numbers[j]
            res_mapping[pdb_number] = i + 1 - skipped
            j += 1

    for line in open(pdb):
        if not line.startswith('ATOM'):
            continue
        type_ = line[13:15]
        chain = line[21:22]
        if type_ not in ('CA', 'CB') or chain != this_chain:
            continue
        resi = res_mapping[int(line[22:26])]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coord = np.array([x, y, z], dtype=np.float32)

        if type_ == 'CA':
            ca[resi] = coord
        else:
            cb[resi] = coord

    # Overwrite CB
    ca.update(cb)

    cmap = dict()
    for i in ca:
        for j in ca:
            if abs(i - j) > 5 and j > i:
                cmap[(i, j)] = np.linalg.norm(ca[i] - ca[j]) 
    return cmap

def parse_contact_matrix(data):
    contacts = dict()
    #new_data = (data + data.T) / 2
    for i in range(data.shape[0] - 4):
        for j in range(i + 5, data.shape[1]):
            contacts[(i + 1, j + 1)] = data[i, j]

    return contacts


def pred_dist(contacts, l_threshold, range_, pdb_parsed):
    pred_single = {}
    pred_fins = {}

    if (n_bins == 1) or (n_bins == 2):
        for (i,j), sc in contacts.items():
            if (sc > 0.5):
                pred_fins[(i,j)] = (sc * thres)
    else: 
        for (i, j), sc in contacts.items():
            temp = (sc[:n_bins])
            pred = [k*l for k, l in zip(temp, bins)]
            sum_prob = 0
            for p in pred:
                sum_prob += p
            pred_fins[(i,j)] = sum_prob
            temp = 0
    #print (pred_fins)

    return (pred_fins)

lengths = dict((line.split(',')[0], int(line.split(',')[1])) for line in open('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/lengths.txt'))

actual_pdb = {}
act_dist_list = []
pred_dist_list = []

for data_file in tqdm.tqdm(glob.glob('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/*.npz'), desc='Protein'):
    data_batch = dict(np.load(data_file))
    data_batch['mask'][:] = 1.

    pred = m.predict(data_batch)[0]
    prot_name = data_file.split('/')[-1].split('.')[0]
    length = lengths[prot_name]
    
    pdb_parsed = parse_pdb('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmarkset/{}/native.pdb'.format(prot_name))
    #print (pdb_parsed)
    contacts_parsed = parse_contact_matrix(pred.squeeze())


    for k, v in pdb_parsed.items():
        if (v < thres):
            actual_pdb[k] = v


    pred_parsed = {}
    pred_parsed = pred_dist(contacts_parsed, threshold_length, range_mode, pdb_parsed)
    
    for (i, j), sc in actual_pdb.items():
        if (i, j) in contacts_parsed.keys():
            print (i,j)
            print (sc)
            print (pred_parsed[(i,j)])
            act_dist_list.append(sc)
            pred_dist_list.append(pred_parsed[(i,j)])

    #print (act_dist_list)
    #print (pred_dist_list)

