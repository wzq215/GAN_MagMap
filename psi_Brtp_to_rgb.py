import matplotlib.pyplot as plt
import numpy as np

from ps_read_hdf_3d import *
from PIL import Image
import tensorflow as tf


def vector_magmap(crid, r_index):
    data_br = ps_read_hdf_3d(crid, 'corona', 'br002')
    data_bt = ps_read_hdf_3d(crid, 'corona', 'bt002')
    data_bp = ps_read_hdf_3d(crid, 'corona', 'bp002')

    r_br = np.array(data_br['scales1'])  # 255 in Rs, distance from sun
    t_br = np.array(data_br['scales2'])  # 150 in rad, latitude
    p_br = np.array(data_br['scales3'])  # 256 in rad, Carrington longitude
    br = np.array(data_br['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
    br_r2 = br * r_br ** 2
    # br = br * 2.205e5  # nT
    # print(r_br.shape)
    # print(t_br)
    # print(p_br)
    # print(br.shape)
    # print(np.nanmin(br*r_br**2,axis=(0,1)))
    # print(np.nanmax(br*r_br**2,axis=(0,1)))
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(r_br)
    # plt.xlabel('index')
    # plt.ylabel('r [Rs]')
    # plt.subplot(1,2,2)
    # plt.plot(np.diff(r_br))
    # plt.xlabel('index')
    # plt.ylabel('delta r [Rs]')
    # plt.suptitle('Corona heights of layers')
    # # plt.plot(abs(np.nanmin(br*r_br**2,axis=(0,1))))
    # # plt.plot(abs(np.nanmax(br*r_br**2,axis=(0,1))))
    # plt.show()

    r_bt = np.array(data_bt['scales1'])  # 201 in Rs, distance from sun
    t_bt = np.array(data_bt['scales2'])  # 150 in rad, latitude
    p_bt = np.array(data_bt['scales3'])  # 256 in rad, Carrington longitude
    bt = np.array(data_bt['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
    bt_r2 = bt * r_bt ** 2
    # bt = bt * 2.205e5  # nT
    # print(r_bt.shape)
    # print(t_bt.shape)
    # print(p_bt.shape)
    # plt.figure()
    # # plt.plot(r_br)
    # plt.plot(abs(np.nanmin(bt*r_bt**2,axis=(0,1))))
    # plt.plot(abs(np.nanmax(bt*r_bt**2,axis=(0,1))))
    # plt.show()

    r_bp = np.array(data_bp['scales1'])  # 201 in Rs, distance from sun
    t_bp = np.array(data_bp['scales2'])  # 150 in rad, latitude
    p_bp = np.array(data_bp['scales3'])  # 256 in rad, Carrington longitude
    bp = np.array(data_bp['datas'])  # 1CU = 2.205G = 2.205e-4T = 2.205e5nT
    bp_r2 = bp * r_bp ** 2
    # bp = bp * 2.205e5  # nT
    # print(r_bp.shape)
    # print(t_bp.shape)
    # print(p_bp.shape)
    # plt.figure()
    # # plt.plot(r_br)
    # plt.plot(abs(np.nanmin(bp*r_bp**2,axis=(0,1))))
    # plt.plot(abs(np.nanmax(bp*r_bp**2,axis=(0,1))))
    # plt.show()

    # plt.figure
    # plt.subplot(3,1,1)
    # plt.suptitle('PSI_r='+str(r_br[r_index])+'Rs')
    # plt.title('Br*r^2')
    # plt.imshow(br_r2[:,:,r_index].T)
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.subplot(3,1,2)
    # plt.imshow(bt_r2[:,:,r_index].T)
    # plt.title('Bt*r^2')
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.subplot(3,1,3)
    # plt.imshow(bp_r2[:,:,r_index].T)
    # plt.title('Bp*r^2')
    # plt.colorbar()
    # plt.ylabel('Pixel [0-142]')
    # plt.xlabel('Pixel [0-299]')
    # plt.gca().invert_yaxis()
    # plt.show()
    x_lon = 299
    y_lat = 142
    new_lon = np.deg2rad(np.linspace(0., 360., 512))
    new_lat = np.deg2rad(np.linspace(0., 180., 256))
    [newLon, newLat] = np.meshgrid(new_lon, new_lat)
    from scipy import interpolate
    # f_br = interpolate.interp2d(p_br,t_br,np.squeeze(br_r2[:,:,r_index]).T)
    # f_bt = interpolate.interp2d(p_bt,t_bt,np.squeeze(bt_r2[:,:,r_index]).T)
    # f_bp = interpolate.interp2d(p_bp,t_bp,np.squeeze(bp_r2[:,:,r_index]).T)
    # brr2_interp = f_br(newLon.ravel(),newLat.ravel()).reshape(360,180)
    # btr2_interp = f_bt(newLon.ravel(),newLat.ravel()).reshape(360,180)
    # bpr2_interp = f_bp(newLon.ravel(),newLat.ravel()).reshape(360,180)
    [pp_br, tt_br] = np.meshgrid(p_br, t_br)
    points = (pp_br.ravel(), tt_br.ravel())
    brr2_interp = interpolate.griddata(points, br_r2[:, :, r_index].T.ravel(), (newLon, newLat))
    [pp_bt, tt_bt] = np.meshgrid(p_bt, t_bt)
    points = (pp_bt.ravel(), tt_bt.ravel())
    btr2_interp = interpolate.griddata(points, bt_r2[:, :, r_index].T.ravel(), (newLon, newLat))
    [pp_bp, tt_bp] = np.meshgrid(p_bp, t_bp)
    points = (pp_bp.ravel(), tt_bp.ravel())
    bpr2_interp = interpolate.griddata(points, bp_r2[:, :, r_index].T.ravel(), (newLon, newLat))
    # norm_Br = 90
    # norm_Bt = 60
    # norm_Bp = 60
    #
    # Br_norm_0 = br[:,:,0]/norm_Br * 255 + 127.5
    #
    # im = Image.new('RGB',(x_lon,y_lat))
    # for i in range(0,x_lon):
    #     for j in range(0,y_lat):
    #         im.putpixel((i,j),(int(br[i,j,0]*)))
    magmap = np.zeros((256, 512, 3))

    magmap[:, :, 0] = brr2_interp
    magmap[:, :, 1] = btr2_interp
    magmap[:, :, 2] = bpr2_interp
    # plt.figure
    # plt.subplot(3,2,1)
    # plt.suptitle('PSI_r='+str(r_br[r_index])+'Rs')
    # plt.title('Br*r^2')
    # plt.pcolormesh(p_br,t_br,br_r2[:,:,r_index].T)
    # plt.colorbar()
    # plt.clim(-25,25)
    # # plt.gca().invert_yaxis()
    # plt.subplot(3,2,3)
    # plt.pcolormesh(p_bt,t_bt,bt_r2[:,:,r_index].T)
    # plt.title('Bt*r^2')
    # plt.colorbar()
    # plt.clim(-20,20)
    # # plt.gca().invert_yaxis()
    # plt.subplot(3,2,5)
    # plt.pcolormesh(p_bp,t_bp,bp_r2[:,:,r_index].T)
    # plt.title('Bp*r^2')
    # plt.colorbar()
    # plt.clim(-20,20)
    # plt.ylabel('lat [rad]')
    # plt.xlabel('lon [rad]')
    # # plt.gca().invert_yaxis()
    # plt.subplot(3,2,2)
    # plt.suptitle('Orgin vs Interpolate_r='+str(r_br[r_index])+'Rs')
    # plt.title('Br*r^2')
    # plt.pcolormesh(new_lon,new_lat,brr2_interp)
    # plt.colorbar()
    # plt.clim(-25,25)
    # # plt.gca().invert_yaxis()
    # plt.subplot(3,2,4)
    # plt.pcolormesh(new_lon,new_lat,btr2_interp)
    # plt.title('Bt*r^2')
    # plt.colorbar()
    # plt.clim(-20,20)
    # # plt.gca().invert_yaxis()
    # plt.subplot(3,2,6)
    # plt.pcolormesh(new_lon,new_lat,bpr2_interp)
    # plt.title('Bp*r^2')
    # plt.colorbar()
    # plt.clim(-20,20)
    # # plt.gca().invert_yaxis()
    # plt.ylabel('lat [rad]')
    # plt.xlabel('lon [rad]')
    # plt.show()
    # quit()
    print('Interp ' + str(crid) + '_' + str(r_index) + '...')
    return magmap


crid = 224
# r_index = 0
path = 'Data/tmp_data_magmap/'
for crid in range(2247, 2263):
    magmap_pre = vector_magmap(crid, 0)
    for r_index in range(0, 10):
        magmap_tmp = vector_magmap(crid, r_index + 1)
        magmap_pair = (tf.convert_to_tensor(magmap_pre), tf.convert_to_tensor(magmap_tmp))
        # print(magmap_pair[0])
        # print(magmap_pair[1])
        print('Save ' + str(crid) + '_' + str(r_index) + '&' + str(r_index + 1) + ' pair')
        np.save(path + 'magmap_' + str(crid) + '_' + str(r_index) + '.npy', magmap_pair)
        magmap_pre = magmap_tmp

# print(np.array(magmap).shape)
#
# magmap = tf.convert_to_tensor(magmap)
# print(magmap)
