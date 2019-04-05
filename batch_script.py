import multiprocessing
import os

cmds = []
for threshold in (0.5, 2):
    for mode in ('all', 'short', 'medium', 'long'):
                cmd = "CUDA_VISIBLE_DEVICES='' python unet_run_test.py 12 8 {} {}".format(threshold, mode)
                cmds.append(cmd)


pool = multiprocessing.Pool(processes=8)
pool.map(os.system, cmds)


