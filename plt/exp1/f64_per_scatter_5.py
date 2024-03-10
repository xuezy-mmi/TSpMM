import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

# 0 matrix 
# 1 nnzA
# 2 dasp_gflops
# 3 cusparse
# 4 csr5
# 5 tile
# 6 bsr
# 7 lsrb
# 8 dasp/cupsarse
# 9 dasp/csr5
# 10 dasp/tile
# 11 dasp/bsr
# 12 dasp/lsrb

tile_com_01=pd.read_csv('spmv_fp64_5_method.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12],names=['compute_nnz','per_dasp','per_cusp','per_csr5','per_tile','per_bsr','per_lsrb','sp_cusp','sp_csr5','sp_tile','sp_bsr','sp_lsrb'])

font = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 22,}
font1 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 22 ,}
font2 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 22,}
font3 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 36 ,}

# fig=plt.figure(figsize=(40,15))
fig, axs = plt.subplots(6, 1, figsize=(17, 22), gridspec_kw={'height_ratios': [1,0.5,0.5,0.5,0.5,0.5]})
#fig=plt.subplot(2,4,gridspec_kw={'height_ratios':[2,1]})
#fig=plt.figure()

plt.subplot(6,1,1)

plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.per_csr5,s=50,c='#f6b654',marker='o',linewidth='0.0',label='CSR5')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.per_tile,s=50,c='#5a6b73',marker='o',linewidth='0.0',label='TileSpMV')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.per_lsrb,s=50,c='#8B864E',marker='o',linewidth='0.0',label='LSRB-CSR')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.per_bsr,s=50,c='#c7dbd5',marker='o',linewidth='0.0',label='cuSPARSE-BSR')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.per_cusp,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='cuSPARSE-CSR')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.per_dasp,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='DASP (this work)')

plt.legend(loc="upper left",fontsize=32,markerscale=1.5)
plt.ylabel("Performance (Gflops)",font3, labelpad=20)
plt.ylim(0,270)
plt.xlim(0,9)
plt.grid(c='grey',alpha=0.8,linestyle='--')
# plt.tick_params(labelsize=24)
plt.tick_params(axis='y',labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(6,1,2)

plt.plot([0, 9], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.sp_csr5,s=50,c='#8c9976',marker='o',linewidth='0.0',label='This work vs. cuSPARSE')
plt.ylim(0,5)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\nCSR5",fontsize=30,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(6,1,3)

plt.plot([0, 9], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.sp_tile,s=50,c='#8c9976',marker='o',linewidth='0.0',label='This work vs. cuSPARSE')
plt.ylim(0,5)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\nTileSpMV",fontsize=30,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(6,1,4)

plt.plot([0, 9], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.sp_lsrb,s=50,c='#8c9976',marker='o',linewidth='0.0',label='This work vs. LSRB-CSR')
plt.ylim(0,8)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\nLSRB-CSR",fontsize=30,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(6,1,5)

plt.plot([0, 9], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.sp_bsr,s=50,c='#8c9976',marker='o',linewidth='0.0',label='This work vs. cuSPARSE-BSR')
plt.ylim(0,8)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\ncuSPARSE-BSR",fontsize=30,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(6,1,6)

plt.plot([0, 9], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(tile_com_01.compute_nnz), tile_com_01.sp_cusp,s=50,c='#8c9976',marker='o',linewidth='0.0',label='This work vs. cuSPARSE-CSR')
plt.ylim(0,4)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\ncuSPARSE-CSR",fontsize=30,labelpad=20)
plt.xlabel("#nonzeros of matrix (log10 scale)",fontsize=32,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=24)




plt.tight_layout()
plt.subplots_adjust(left = 0.14, right = 0.99, wspace = 0.1, hspace= 0.1)

plt.savefig('dasp_f64_5.pdf',dpi=200)

# plt.show()
