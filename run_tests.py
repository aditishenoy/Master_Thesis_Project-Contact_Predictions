import os
import glob
import functools

from keras.models import load_model
import numpy as np
from Bio import pairwise2

import pylab as plt

import tqdm


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
                cmap[(i, j)] = np.linalg.norm(ca[i] - ca[j]) < 8
    return cmap

def compute_ppv(contacts, l_threshold, range_, pdb_parsed):

    low, hi = dict(short=(5, 12), medium=(12, 23), long=(23, 10000), all=(5, 100000))[range_]
    contact_list = [(i, j, sc) for (i, j), sc in contacts.items() if low < j - i <= hi]
    contact_list.sort(key=lambda x: x[2], reverse=True)
    selected = int(round(l_threshold * max(max(k) for k in contacts)))
    contact_list = contact_list[:selected]
    #print (contact_list)

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

lengths = dict((line.split(',')[0], int(line.split(',')[1])) for line in open('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/lengths.txt'))

model_name = 'Mplus_AltOrd_mae_trained'
model_n = 'Mplus_AltOrd'
m = load_model('{}.h5'.format(model_name))
out_f = 'results_0.5_{}.log'.format(model_name)

print()
print(out_f)
print()
output = open(out_f, 'w')


for epoch in tqdm.trange(1, 100, desc='Epoch'):
	'''
	if 'pconsc4' in model_name:
		weights_path = 'models/{}/*_epo{:02d}-*.h5'.format(model_name, epoch)
	else:		
		weights_path = 'models/{}/{}_epo{:02d}-*.h5'.format(model_name, model_name, epoch)
	try:
		weights = glob.glob(weights_path)[0]
	except:
		continue 
        '''

	weights_path = 'models/{}/{}_epo{:02d}-*.h5'.format(model_name, model_n, epoch)

	print (weights_path)
	weights = glob.glob(weights_path)[0]

	
	m.load_weights(weights)
	
	ppv = []

	for data_file in tqdm.tqdm(glob.glob('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmark_set/*.npz'), desc='Protein'):
		data_batch = dict(np.load(data_file))
		data_batch['mask'][:] = 1.
		pred = m.predict(data_batch)[2]
	
		prot_name = data_file.split('/')[-1].split('.')[0]
		length = lengths[prot_name]
		pdb_parsed = parse_pdb('/home/ashenoy/ashenoy/david_retrain_pconsc4/testing/benchmarkset/{}/native.pdb'.format(prot_name))
			
		contacts_parsed = parse_contact_matrix(pred.squeeze())
		this_ppv = compute_ppv(contacts_parsed, 0.5, 'all', pdb_parsed)
		ppv.append(this_ppv)
		#print (epoch, contacts_parsed)
		'''
		plt.imshow(pred.squeeze())
		plt.title('{} : {:2d}'.format(prot_name, epoch))
		plt.savefig('plots/{}_{}_{:02d}.png'.format(model_name, prot_name, epoch))
		plt.close()
		'''
		output = open(out_f, 'w')
		print(epoch, np.mean(ppv), np.median(ppv), file=output, flush=True)
	
		print()
		print()
		output.close()
		
#os.system('cat results_*')
