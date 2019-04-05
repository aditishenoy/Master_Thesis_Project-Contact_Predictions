from __future__ import division

'-------------------------------------------------------------------'
import os
import sys
import glob
import time
import functools

from keras.models import load_model
import numpy as np
np.set_printoptions(threshold=np.nan)

from Bio import pairwise2

import pylab as plt

import tqdm

three_to_one = {'ASP': 'D', 'GLU': 'E', 'ASN': 'N', 'GLN': 'Q', 'ARG': 'R', 'LYS': 'K', 'PRO': 'P', 'GLY': 'G',
                'CYS': 'C', 'THR': 'T', 'SER': 'S', 'MET': 'M', 'TRP': 'W', 'PHE': 'F', 'TYR': 'Y', 'HIS': 'H',
                'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'MSE': 'M'}


model_name = 'M12_R05_D01_Test_Epochs_mae_trained'
model_n = 'M12_R05_D01_Test_Epochs'

#Number of bins used for classification (Based on model_name)
n_bins = int(sys.argv[1])
#Distance threshold to calculate all the other measures (8 or 15)
thres = int(sys.argv[2])
#The value n which is multiplied with L (Length of protein) to get the top n*L contacts
threshold_length = float(sys.argv[3])
#What type of protein length is it (short, medium, long, all)
range_mode = sys.argv[4]

if n_bins == 7:
    bins = [2, 5, 7, 9, 11, 13, 15]
    prob_len = 3

elif n_bins == 26:
    bins = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]
    prob_len = 9

elif n_bins == 12:
    bins = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
    prob_len = 4

else:

    raise ValueError('Invalid number of bins (n_bins): {}'.format(n_bins))


assert 0 < threshold_length < 4., 'Invalid threshold_length to contact top contacts'

assert range_mode in ('short', 'medium', 'long', 'all'), range_mode


'-------------------------------------------------------------------'
# Support to build functions

def _strip(x):
    return len(x[1].strip('-'))

def pick_best_alignment(align, sequence):
    align_1 = [al for al in align if al[0] == sequence]
    if align_1:
        if len(align_1) == 1:
            al = align_1[0]
        else:
            al = min(align_1, key=_strip)
    else:
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

'-------------------------------------------------------------------'
#Performance metrics (absolute error, relative error)

def error_metrics(contacts, l_threshold, range_, pdb_parsed):
    actual_pdb = {}
    pred_contacts = {}
    pred_zipped = {}
    pred_single = {}
    pred_fins = {}
    
    for k, v in pdb_parsed.items():
        if (v < thres):
            actual_pdb[k] = v
    
    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    
    for (i,j), sc in contacts.items():
        if low < (j-i) < hi:
            temp = (sc[:prob_len])
            sum_prob = 0
            for k in temp:
                sum_prob += k
            pred_single[(i,j)] = sum_prob
            temp = 0
    
    selected = int(round(l_threshold * max(max(k) for k in pred_single)))
    sorted_x = sorted(pred_single.items(), key=lambda kv: kv[1], reverse = True)
    #print (sorted_x)
    contact_list = sorted_x[:selected]

    contact_dict = {}
    contact_dict =  dict(((i,j), y) for (i,j), y in contact_list)

    for (i, j), sc in contacts.items():
        if (i, j) in contact_dict.keys():
                temp = (sc[:n_bins])
                pred = [k*l for k, l in zip(temp, bins)]
                #print (pred)
                sum_prob = 0
                for p in pred:
                    sum_prob += p

                pred_fins[(i,j)] = sum_prob
                temp = 0
    
    #print (pred_fins)  
    abs_error = []
    rel_error = []

    for (k, v) in (pred_fins.items()):
        for (k1, v1) in pdb_parsed.items():
            if (k == k1):
                #print (k)
                #print (k1)
                #print (v)
                #print (v1)  
                abs_error.append(abs(v - v1))
                rel_error.append((abs(v-v1))/((v+v1)/2)) 
    fin = 0
    fins = 0
    if (len(abs_error) > 0):
        fin = np.mean(abs_error)
    if (len(rel_error) > 0):
        fins = np.mean(rel_error)
    
    return fin, fins


# Performance metrics (Precision)
def alt_metrics_p(contacts, l_threshold, range_, pdb_parsed):
    actual_pdb = {}
    pred_contacts = {}
    pred_zipped = {}
    pred_single = {}
    pred_fins = {}

    for k, v in pdb_parsed.items():
        if (v < thres):
            actual_pdb[k] = v

    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    for (i,j), sc in contacts.items():
        if low < (j-i) < hi:
            temp = (sc[:prob_len])
            sum_prob = 0
            for k in temp:
                sum_prob += k
            pred_single[(i,j)] = sum_prob
    
    selected = int(round(l_threshold * max(max(k) for k in pred_single)))
    sorted_x = sorted(pred_single.items(), key=lambda kv: kv[1], reverse = True)
    #print (sorted_x)
    contact_list = sorted_x[:selected]

    contact_dict = {}
    contact_dict =  dict(((i,j), y) for (i,j), y in contact_list)

    for (i, j), sc in contacts.items():
        if (i, j) in contact_dict.keys():
                temp = (sc[:n_bins])
                pred = [k*l for k, l in zip(temp, bins)]
                #print (pred)
                sum_prob = 0
                for p in pred:
                    sum_prob += p
                pred_fins[(i,j)] = sum_prob
                temp = 0
    
    count = 0
    tot_count = 0 

    for k, v in pred_fins.items():
        if (v < thres):
            pred_contacts[k] = v

    for (i, j) in pred_contacts.keys():
        if (i, j) in actual_pdb.keys():
            count += 1
        tot_count += 1
        
    if tot_count == 0:
        tot_count = 1

    #print (count/tot_count)
    return (count/tot_count)

# Performance metrics (Recall)
def alt_metrics_r(contacts, l_threshold, range_, pdb_parsed):
    actual_pdb = {}
    pred_contacts = {}
    pred_zipped = {}
    pred_single = {}
    pred_fins = {}

    for k, v in pdb_parsed.items():
        if (v < thres):
            actual_pdb[k] = v

    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    for (i,j), sc in contacts.items():
        if low < (j-i) < hi:
            temp = (sc[:prob_len])
            sum_prob = 0
            for k in temp:
                sum_prob += k
            pred_single[(i,j)] = sum_prob
    
    selected = int(round(l_threshold * max(max(k) for k in pred_single)))
    sorted_x = sorted(pred_single.items(), key=lambda kv: kv[1], reverse = True)
    #print (sorted_x)
    contact_list = sorted_x[:selected]

    contact_dict = {}
    contact_dict =  dict(((i,j), y) for (i,j), y in contact_list)

    for (i, j), sc in contacts.items():
        if (i, j) in contact_dict.keys():
                temp = (sc[:n_bins])
                pred = [k*l for k, l in zip(temp, bins)]
                #print (pred)
                sum_prob = 0
                for p in pred:
                    sum_prob += p
                pred_fins[(i,j)] = sum_prob
                temp = 0
    
    count = 0
    tot_count = 0 

    for k, v in pred_fins.items():
        if (v < thres):
            pred_contacts[k] = v

    for (i, j) in actual_pdb.keys():
        if (i, j) in pred_contacts.keys():
            count += 1
        tot_count += 1
        
    if tot_count == 0:
        tot_count = 1
    #print (count/tot_count)
    return (count/tot_count)

'-------------------------------------------------------------------'

#Main part of the program which call associated functions

lengths = dict((line.split(',')[0], int(line.split(',')[1])) for line in open('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/lengths.txt'))

m = load_model('{}.h5'.format(model_name))

out_pm = 'results_{}_{}'.format(range_mode, model_name)
print()
print(out_pm)
print()
output = open(out_pm, 'w')

for epoch in tqdm.trange(1, 51, desc = 'Epoch'):

    weights_path = 'models/{}/{}_epo{:02d}-*.h5'.format(model_name, model_n, epoch)

    weights = glob.glob(weights_path)[0]

    m.load_weights(weights)

    #ppv = []

    abb = []
    rell = []
    precc = []
    rec = []
    f1_s = []

    #t_predict = 0.
    #t_parsing = 0.
    #t_compute_error = 0.
    #t_metrics = 0.

    for data_file in tqdm.tqdm(glob.glob('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/*.npz'), desc='Protein'):
        data_batch = dict(np.load(data_file))
        data_batch['mask'][:] = 1

        #t0 = time.time()
        pred = m.predict(data_batch)[0]
        #t_predict += time.time() - t0
        #print (pred)
        print (pred.shape)
        
        prot_name = data_file.split('/')[-1].split('.')[0]
        length = lengths[prot_name]
        
        #t0 = time.time()
        pdb_parsed = parse_pdb('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmarkset/{}/native.pdb'.format(prot_name))
        contacts_parsed = parse_contact_matrix(pred.squeeze())
        #t_parsing += time.time() - t0
       
        #t0 = time.time()
        ab_error, rel_error = error_metrics(contacts_parsed, threshold_length, range_mode,  pdb_parsed)
	#t_compute_error += time.time() - t0

        #t0 = time.time()
        prec = alt_metrics_p(contacts_parsed, threshold_length, range_mode, pdb_parsed)
        recall = alt_metrics_r(contacts_parsed, threshold_length, range_mode, pdb_parsed)
        #t_metrics += time.time() - t0

        if (ab_error != 0):
            abb.append(ab_error)
        
        if (rel_error != 0):
            rell.append(rel_error)
        
        if (prec != 0) and (recall != 0):
            f1 = ((2*prec*recall)/(prec+recall))
            f1_s.append(f1)

        if (prec != 0):
            precc.append(prec)

        if (recall != 0):
            rec.append(recall)

        #print (precc)
        #print (rec)
        #print (f1_s)

        #Save metrics to file
        output = open(out_pm, 'w')
        print(epoch, np.mean(abb), np.median(abb), np.mean(rell), np.median(rell), np.mean(precc), np.mean(rec), np.mean(f1_s), file=output, flush=True)
        print()
        print()
        output.close()

#print('Time spent on predictions:',    t_predict)
#print('Time spent computing errors:',    t_compute_error)
#print('Time spent computing metrics:',    t_metrics )
#print('Time parsing:', t_parsing)


#os.system('cat results_*')

'-------------------------------------------------------------------'


