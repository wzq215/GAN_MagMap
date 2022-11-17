import numpy as np
import requests
import pandas as pd


def download_ips_data(year):
    try:
        r = requests.get('http://stsw1.isee.nagoya-u.ac.jp/vlist/map/Vannual' + str(year) + '.dat')
        with open('Data/IPS_Data/Vannual' + str(year) + '.dat', 'wb') as f:
            f.write(r.content)
        print('Downloading IPS data for year ' + str(year))
    except Exception as e:
        print(e.args)
        print(str(e))
        print(repr(e))
        print('Nothing Found for year ' + str(year))
    return 0


def load_ips_data_to_npy(year):
    try:
        data = pd.read_csv('Data/IPS_Data/Vannual' + str(year) + '.dat')
        ips_map = np.zeros((360, 180, 11))
        iline = 0
        for lat in range(180):
            for cr in range(11):
                for lon in range(360):
                    if iline < data.size:
                        ips_map[lon, lat, cr] = data.values[iline]
                        iline += 1
        year_cr = pd.read_table('Data/IPS_Data/start_CR.txt', header=None)
        start_cr = year_cr[year_cr[0] == year][1].values[0]
        print(year, start_cr)
        for cr in range(11):
            np.save('Data/IPS_Data/V_map/cr' + str(start_cr + cr) + '.npy', ips_map[:, :, 10 - cr])
            print('Saving CR' + str(start_cr + cr) + '. NaN counts: ', np.sum(ips_map[:, :, 10 - cr] == 0))
            with open('Data/IPS_Data/V_map/count_nans.txt', 'a') as f:
                f.write(str(start_cr + cr) + ' ')
                f.write(str(np.sum(ips_map[:, :, cr] == 0)) + '\n')
    except Exception as e:
        print(e.args)
        print(str(e))
        print(repr(e))
        print('Nothing Found for year ' + str(year))
    return 0


for year in range(1985, 2021):
    download_ips_data(year)
    load_ips_data_to_npy(year)
