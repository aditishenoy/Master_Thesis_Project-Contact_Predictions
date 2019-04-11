import matplotlib.pyplot as plt
import numpy as np
'''


fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

fig.tight_layout()
'''

num_bins = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
x = [0.00097076, 0.00169707, 0.00392179, 0.00421804, 0.00774152, 0.00953522, 0.01490338, 0.0169697,  0.02030931, 0.02606114, 0.03390031, 0.85977185]
#plt.hist(x, color = 'blue', edgecolor = 'black', bins = num_bins)
plt.plot(x)
plt.show()




y= [0.00334347, 0.00404495, 0.00225471, 0.00218266, 0.00208441, 0.00228486, 0.00278001, 0.00297202, 0.00446296, 0.00528383, 0.00743773, 0.9608684 ]
plt.hist(y, color = 'blue', edgecolor = 'black', bins = num_bins)
plt.show()


