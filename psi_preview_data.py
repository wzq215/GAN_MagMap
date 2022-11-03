import numpy as np
from matplotlib import pyplot as plt

PATH = 'Data/tmp_data_magmap/train/'
count = 0
crids = np.arange(1625, 2050)
br_min_1s = crids * 0.
br_max_1s = crids * 0.
br_min_45s = crids * 0.
br_max_45s = crids * 0.
bt_min_1s = crids * 0.
bt_max_1s = crids * 0.
bt_min_45s = crids * 0.
bt_max_45s = crids * 0.
bp_min_1s = crids * 0.
bp_max_1s = crids * 0.
bp_min_45s = crids * 0.
bp_max_45s = crids * 0.

for crid in crids:
    try:
        tmp_data = np.load(PATH + 'magmap_' + str(crid) + '_1_45.npy')
        print(tmp_data)
    except:
        print('No Data For CR' + str(crid))
        continue
    count += 1
    magmap_1 = np.squeeze(tmp_data[0, :, :, :])
    magmap_45 = np.squeeze(tmp_data[1, :, :, :])

    br_min_1s[crid - 1625] = np.nanmin(magmap_1[:, :, 0])
    br_max_1s[crid - 1625] = np.nanmax(magmap_1[:, :, 0])
    br_min_45s[crid - 1625] = np.nanmin(magmap_45[:, :, 0])
    br_max_45s[crid - 1625] = np.nanmax(magmap_45[:, :, 0])

    bt_min_1s[crid - 1625] = np.nanmin(magmap_1[:, :, 1])
    bt_max_1s[crid - 1625] = np.nanmax(magmap_1[:, :, 1])
    bt_min_45s[crid - 1625] = np.nanmin(magmap_45[:, :, 1])
    bt_max_45s[crid - 1625] = np.nanmax(magmap_45[:, :, 1])

    bp_min_1s[crid - 1625] = np.nanmin(magmap_1[:, :, 2])
    bp_max_1s[crid - 1625] = np.nanmax(magmap_1[:, :, 2])
    bp_min_45s[crid - 1625] = np.nanmin(magmap_45[:, :, 2])
    bp_max_45s[crid - 1625] = np.nanmax(magmap_45[:, :, 2])
    # print(min_1s[crid-1625],max_1s[crid-1625],min_45s[crid-1625],max_45s[crid-1625])
    # plt.imshow(magmap_45)
    # plt.show()
print(count)
plt.figure
plt.subplot(3, 1, 1)
plt.title('max/min Br (r=1.0Rs)')
plt.plot(crids, br_min_1s)
plt.plot(crids, br_max_1s)
plt.ylim([-60, 60])
plt.subplot(3, 1, 2)
plt.title('max/min Br (r=2.5Rs)')
plt.plot(crids, br_min_45s)
plt.plot(crids, br_max_45s)
plt.ylim([-.5, .5])
plt.subplot(3, 1, 3)
plt.title('|max-min|(r=2.5Rs)/|max-min|(r=1.0Rs) [%]')
plt.plot(crids, (br_max_45s - br_min_45s) / (br_max_1s - br_min_1s) * 100)
plt.ylim([0, 5])
plt.xlabel('CRID')
plt.show()

plt.figure
plt.subplot(3, 1, 1)
plt.title('max/min Br (r=1.0Rs)')
plt.plot(crids, bt_min_1s)
plt.plot(crids, bt_max_1s)
plt.ylim([-25, 25])
plt.subplot(3, 1, 2)
plt.title('max/min Br (r=2.5Rs)')
plt.plot(crids, bt_min_45s)
plt.plot(crids, bt_max_45s)
plt.ylim([-.1, .1])
plt.subplot(3, 1, 3)
plt.title('|max-min|(r=2.5Rs)/|max-min|(r=1.0Rs) [%]')
plt.plot(crids, (bt_max_45s - bt_min_45s) / (bt_max_1s - bt_min_1s) * 100)
plt.ylim([0, .5])
plt.xlabel('CRID')
plt.show()

plt.figure
plt.subplot(3, 1, 1)
plt.title('max/min Br (r=1.0Rs)')
plt.plot(crids, bp_min_1s)
plt.plot(crids, bp_max_1s)
plt.ylim([-30, 30])
plt.subplot(3, 1, 2)
plt.title('max/min Br (r=2.5Rs)')
plt.plot(crids, bp_min_45s)
plt.plot(crids, bp_max_45s)
plt.ylim([-.05, .05])
plt.subplot(3, 1, 3)
plt.title('|max-min|(r=2.5Rs)/|max-min|(r=1.0Rs) [%]')
plt.plot(crids, (bp_max_45s - bp_min_45s) / (bp_max_1s - bp_min_1s) * 100)
plt.ylim([0, .5])
plt.xlabel('CRID')
plt.show()
