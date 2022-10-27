# GAN_MagMap

Course project For space weather forcasting.

* Goal: To construct the magnetic field in the inner heliosphere based on machine learning.

## TODO LIST

* Customize input data for pix2pix algorithm.

## Notes about data input.

CORHEL MHD data is used in this project as input. There are two types of PSI data:

* Medium: cr1625 to cr2240. Total 15 CRs. Not provided in the future. |
* High: cr2235 to cr2262. Total 28 CRs. Higher resolution & New codes. | Too Few!!!

Note that each High-CR has at least three sets of data: hmi_masp_mas_std_0201; hmi_mast_mas_std_0101;
hmi_mast_mas_std_0201. Medium CRs may use different MHD algorithms.

  
