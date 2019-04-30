# Regression script
# Calculates Precision, Recall (TPR), F1 Score, Specificity, False Positive Rate
'-------------------------------------------------------------------'

import os
import glob
import functools
import os
import sys

import time
import functools
from keras.models import load_model
import numpy as np
from Bio import pairwise2

import pylab as plt

import tqdm

n_bins = int(sys.argv[1])

if n_bins == 1:
    model_name = 'model12_mae_trained'
    model_n = 'model12'
    selected_epoch = 14

elif n_bins == 2:
    model_name = 'Mplus_AltRegDouble12_mae_trained'
    model_n = 'Mplus_AltRegDouble12'
    selected_epoch = 75


#Distance threshold to calculate all the other measures (8 or 15)
thres = int(sys.argv[2])

#The value n which is multiplied with L (Length of protein) to get the top n*L contacts
threshold_length = int(sys.argv[3])


range_mode = (sys.argv[4])

assert 0 < threshold_length < 4., 'Invalid threshold_length to contact top contacts'

assert range_mode in ('short', 'medium', 'long', 'all'), range_mode

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
            skipped = aligned[0][:j].count('-')
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
                cmap[(i, j)] = np.linalg.norm(ca[i] - ca[j]) < thres 
    return cmap

'-------------------------------------------------------------------'
# Metrics
def compute_ppv(contacts, l_threshold, range_, pdb_parsed):

    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    contact_list = [(i, j, sc) for (i, j), sc in contacts.items() if low < j - i <= hi]
    contact_list.sort(key=lambda x: x[2], reverse=True)
    selected = int(round(l_threshold * max(max(k) for k in contacts)))
    contact_list = contact_list[:selected]
    
    pos_counter = 0
    total_counter = 0
    for i, j, _ in contact_list:
        if (i, j) in pdb_parsed:
            if pdb_parsed[(i, j)]:
                pos_counter += 1
            total_counter += 1

    if total_counter == 0:
        total_counter = 1
    return pos_counter / total_counter
    


def tpr_calc(contacts, l_threshold, range_, pdb_parsed):
    actual_pdb = {}

    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    contact_list = [(i, j, sc) for (i, j), sc in contacts.items() if low < j - i <= hi]
    contact_list.sort(key=lambda x: x[2], reverse=True)
    selected = int(round(l_threshold * max(max(k) for k in contacts)))
    contact_list = contact_list[:selected]
    
    contact_dict = {}
    contact_dict =  dict(((i,j), y) for i, j, y in contact_list)

    count = 0
    tot_count = 0 

    for (i, j) in pdb_parsed.keys():
        if pdb_parsed[(i,j)]:
            if (i, j) in contact_dict.keys():
                count += 1
            tot_count += 1
        
    if tot_count == 0:
        tot_count = 1
    
    return (count/tot_count)

def fpr_calc(contacts, l_threshold, range_, pdb_parsed):
    actual_pdb = {}

    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    contact_list = [(i, j, sc) for (i, j), sc in contacts.items() if low < j - i <= hi]
    contact_list.sort(key=lambda x: x[2], reverse=True)
    selected = int(round(l_threshold * max(max(k) for k in contacts)))
    contact_list = contact_list[:selected]
    
    contact_dict = {}
    contact_dict =  dict(((i,j), y) for i, j, y in contact_list)

    count = 0
    tot_count = 0 

    for (i, j) in pdb_parsed.keys():
        if not (pdb_parsed[(i,j)]):
            if (i, j) in contact_dict.keys():
                count += 1
            tot_count += 1
        
    if tot_count == 0:
        tot_count = 1
    
    return (count/tot_count)

'-------------------------------------------------------------------'

def parse_contact_matrix(data):
    contacts = dict()
    new_data = (data + data.T) / 2
    for i in range(data.shape[0] - 4):
        for j in range(i + 5, data.shape[1]):
            contacts[(i + 1, j + 1)] = new_data[i, j]  

    return contacts

def to_image(cmap, N):
    data = np.zeros((N, N))
    for (i,j), v in cmap.items():
        if v:
            data[i-1, j-1] = 1

    return data

'-------------------------------------------------------------------'
lengths = dict((line.split(',')[0], int(line.split(',')[1])) for line in open('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/lengths.txt'))

m = load_model('{}.h5'.format(model_name))
weights_path = 'regression/models/{}/{}_epo{:02d}-*.h5'.format(model_name, model_n, selected_epoch)
weights = glob.glob(weights_path)[0]
m.load_weights(weights)

out_pm = 'results/results_{}_{}_{}'.format(thres, range_mode, model_name)
print()
print(out_pm)
print()
output = open(out_pm, 'w')

ppv = []
rec = []
f1_s = []

spe = []
fpr = []

for data_file in tqdm.tqdm(glob.glob('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/*.npz'), desc='Protein'):
    data_batch = dict(np.load(data_file))
    data_batch['mask'][:] = 1.

    pred = m.predict(data_batch)[0]
    prot_name = data_file.split('/')[-1].split('.')[0]
    length = lengths[prot_name]

    pdb_parsed = parse_pdb('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmarkset/{}/native.pdb'.format(prot_name))
    contacts_parsed = parse_contact_matrix(pred.squeeze())

    prec = compute_ppv(contacts_parsed, threshold_length, range_mode, pdb_parsed)
    recall = tpr_calc(contacts_parsed, threshold_length, range_mode, pdb_parsed)
    ppv.append(prec)
    rec.append(recall)

    if (prec != 0) and (recall != 0):
            f1 = ((2*prec*recall)/(prec+recall))
            f1_s.append(f1)

    fp_rate = fpr_calc(contacts_parsed, threshold_length, range_mode, pdb_parsed)
    spe.append(fp_rate)
    fpr.append(1-fp_rate)

#Save metrics to file
output = open(out_pm, 'w')
print(np.mean(ppv), np.mean(rec), np.mean(f1_s), np.mean(spe), np.mean(fpr), file=output, flush=True)
print()
print()
output.close()

'-------------------------------------------------------------------'
