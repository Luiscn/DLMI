import numpy as np

pat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
          207, 208, 209, 210, 211, 212, 213]

pat_elems = [1, 14, 98, 256, 500, 600, 700, 698, 808, 969, 468, 369, 147,
             345, 402, 696, 890, 702, 8, 90, 42, 69, 89, 72]

distComp = np.zeros((2, len(pat_ids)))

for ii in range(len(pat_ids)):
    const_pat_id, const_elem_id = pat_ids[ii], pat_elems[ii]
    runfile('/Volumes/My Passport/baseCompare.py', wdir='/Volumes/My Passport')
    distComp[0,ii] = sDist(true_mag, nn_mag)
    distComp[1,ii] = sDist(true_mag, mean_mag)
 
# create plot
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(len(pat_ids))
bar_width = 0.3
rects1 = plt.bar(index, distComp[0,:], bar_width, label='true v pred')
rects2 = plt.bar(index + bar_width, distComp[1,:], bar_width, label='true v mean')

plt.xlabel('Patients')
plt.ylabel('Dist')
plt.title('Dists, less blue is better')
plt.xticks(index + bar_width/2, pat_ids)
plt.legend()
plt.tight_layout()
plt.show()