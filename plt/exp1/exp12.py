import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker
import seaborn
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

# tile_com_01=pd.read_csv('spmv_fp64_5_method.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12],names=['compute_nnz','per_dasp','per_cusp','per_csr5','per_tile','per_bsr','per_lsrb','sp_cusp','sp_csr5','sp_tile','sp_bsr','sp_lsrb'])

csvfile1=pd.read_csv('all-zero_col_num_8.csv',usecols=[1,2,3,4,5,6,7],names=["ROW","COL","NNZ","cnt0_8","cnt1_8","new_sparsity_ratio1","new_sparsity_ratio2"])
csvfile2=pd.read_csv('all-zero_col_num_16.csv',usecols=[1,2,3,4,5,6,7],names=["ROW","COL","NNZ","cnt0_8","cnt1_8","new_sparsity_ratio1","new_sparsity_ratio2"])


sparsity = []
ROW = []
COL = []
NNZ = []
cnt0_8 = []
cnt1_8 = []
cnt0_16 = []
cnt1_16 = []
new_sparsity_ratio1 = []
new_sparsity_ratio2 = []
new_sparsity_ratio1_16 = []
new_sparsity_ratio2_16 = []
TC8_utilization1 = []
TC8_utilization2 = []
TC16_utilization1 = []
TC16_utilization2 = []
matrix_num = len(csvfile1.ROW)-1
for i in range(matrix_num):
    ROW.append(int(csvfile1.ROW[i+1]))
    COL.append(int(csvfile1.COL[i+1]))
    NNZ.append(int(csvfile1.NNZ[i+1]))
    row = int(csvfile1.ROW[i+1])
    col = int(csvfile1.COL[i+1])
    nnz = int(csvfile1.NNZ[i+1])
    cnt0_8 = int(csvfile1.cnt0_8[i+1])
    cnt1_8 = int(csvfile1.cnt1_8[i+1])
    cnt0_16 = int(csvfile2.cnt0_8[i+1])
    cnt1_16 = int(csvfile2.cnt1_8[i+1])
    num0_vec8 = int(row*col/8)-cnt0_8
    num1_vec8 = int(row*col/8)-cnt1_8
    num0_vec16 = int(row*col/16)-cnt0_16
    num1_vec16 = int(row*col/16)-cnt1_16
    if(nnz == 0):
        u0 = 0.0
        u1 = 0.0
        u2 = 0.0
        u3 = 0.0
    else:
        u0 = nnz / (num0_vec8 * 8)
        u1 = nnz / (num1_vec8 * 8)
        u2 = nnz / (num0_vec16 * 16)
        u3 = nnz / (num1_vec16 * 16)
    # TC8_utilization1.append(1.0 - float(csvfile1.new_sparsity_ratio1[i+1]))
    # TC8_utilization2.append(1.0 - float(csvfile1.new_sparsity_ratio2[i+1]))
    # TC16_utilization1.append(1.0 - float(csvfile2.new_sparsity_ratio1[i+1]))
    # TC16_utilization2.append(1.0 - float(csvfile2.new_sparsity_ratio2[i+1]))
    TC8_utilization1.append(u0)
    TC8_utilization2.append(u1)
    TC16_utilization1.append(u2)
    TC16_utilization2.append(u3)

for i in range(matrix_num):
    ratio = 1.0 - NNZ[i]/(ROW[i]*COL[i])
    sparsity.append(ratio)
# print(TC8_utilization1)
# print(TC8_utilization2)


font = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 36,}
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 36 ,}
font2 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 36,}
font3 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 20 ,}

# fig=plt.figure(figsize=(40,15))
# fig, axs = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1,1]})
#fig=plt.subplot(2,4,gridspec_kw={'height_ratios':[2,1]})
#fig=plt.figure()
fig, axs = plt.subplots(1, 1, figsize=(16, 8))
# plt.subplot(
plt.scatter(sparsity, TC16_utilization1,s=50,c='#81d8cf',marker='o',linewidth='0.0',label='V = 16')
plt.scatter(sparsity, TC16_utilization2,s=50,c='#ffb4c8',marker='o',linewidth='0.0',label='V = 16 (reordering)')
plt.scatter(sparsity, TC8_utilization1,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='V = 8')
plt.scatter(sparsity, TC8_utilization2,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='V = 8 (reordering)')

x = np.linspace(0.5, 1, 1000)
y = 1 - x
plt.plot(x, y,c='black')

plt.legend(loc="upper left",fontsize=24,markerscale=1.5)
# plt.ylabel("number of all-zero vector",font3, labelpad=10)
plt.ylim(0,0.6)
# plt.yticks(range(0, 140001, 10000))
plt.xlim(0.5,1)
plt.xlabel("Sparsity",fontsize=24,labelpad=24)
plt.grid(c='grey',alpha=0.8,linestyle='--')
# plt.tick_params(labelsize=24)
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)

# plt.subplot(2,1,2)

# # plt.plot([0, 1], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
# plt.scatter(sparsity, TC16_utilization1,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='Utilization of TC(M=16) (before reordering)')
# plt.scatter(sparsity, TC16_utilization2,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='Utilization of TC(M=16) (after reordering)')
# plt.legend(loc="upper left",fontsize=20,markerscale=1.5)
# # plt.ylabel("number of all-zero vector",font3, labelpad=20)
# plt.ylim(0,0.6)
# # plt.yticks(range(0, 70001, 10000))
# plt.xlim(0.5,1)
# plt.xlabel("#sparsity",fontsize=20,labelpad=20)
# plt.grid(c='grey',alpha=0.8,linestyle='--')
# plt.tick_params(axis='y',labelsize=15)
# plt.tick_params(axis='x',labelsize=15)
plt.tight_layout()
plt.subplots_adjust(left = 0.1, right = 0.98, wspace = 0.1, hspace= 0.1)
fig.text(0.01, 0.5, 'Density in CVSE format', va='center', rotation='vertical', fontsize = 24)


plt.savefig('exp1-2.pdf',dpi=2000)

