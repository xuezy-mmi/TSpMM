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
nnz_num = 0
storage_num = 0
nonzero80 = 0
nonzero81 = 0
nonzero160 = 0
nonzero161 = 0
matrix_num = len(csvfile1.ROW)-1
for i in range(matrix_num):
    ROW.append(int(csvfile1.ROW[i+1]))
    COL.append(int(csvfile1.COL[i+1]))
    NNZ.append(int(csvfile1.NNZ[i+1]))
    cnt0_8.append(int(csvfile1.cnt0_8[i+1]))
    cnt1_8.append(int(csvfile1.cnt1_8[i+1]))
    cnt0_16.append(int(csvfile2.cnt0_8[i+1]))
    cnt1_16.append(int(csvfile2.cnt1_8[i+1]))
    new_sparsity_ratio1.append(float(csvfile1.new_sparsity_ratio1[i+1]))
    new_sparsity_ratio2.append(float(csvfile1.new_sparsity_ratio2[i+1]))
    new_sparsity_ratio1_16.append(float(csvfile2.new_sparsity_ratio1[i+1]))
    new_sparsity_ratio2_16.append(float(csvfile2.new_sparsity_ratio2[i+1]))
    num_vec_8 = int((int(csvfile1.ROW[i+1]))*(int(csvfile1.COL[i+1])) / 8)
    num_vec_16 = int((int(csvfile1.ROW[i+1]))*(int(csvfile1.COL[i+1])) / 16)
    nnz_num += int(csvfile1.NNZ[i+1])
    storage_num += (int(csvfile1.ROW[i+1]))*(int(csvfile1.COL[i+1]))
    nonzero80 += (num_vec_8 - (int(csvfile1.cnt0_8[i+1])))
    nonzero81 += (num_vec_8 - (int(csvfile1.cnt1_8[i+1])))
    nonzero160 += (num_vec_16 - (int(csvfile2.cnt0_8[i+1])))
    nonzero161 += (num_vec_16 - (int(csvfile2.cnt1_8[i+1])))
for i in range(matrix_num):
    ratio = 1.0 - NNZ[i]/(ROW[i]*COL[i])
    sparsity.append(ratio)
# print(cnt1_16)

print(nnz_num)
print(storage_num)
print(nonzero80*8)
print(nonzero81*8)
print(nonzero160*16)
print(nonzero161*16)
loss80 = nonzero80*8 / nnz_num
loss81 = nonzero81*8 / nnz_num
loss160 = nonzero160*16 / nnz_num
loss161 = nonzero161*16 / nnz_num
save80 = storage_num / (nonzero80*8)
save81 = storage_num / (nonzero81*8)
save160 = storage_num / (nonzero160*16)
save161 = storage_num / (nonzero161*16)
reorder8 = nonzero80/nonzero81
reorder16 = nonzero160/nonzero161
print("storage loss to csr")
print(loss80)
print(loss81)
print(loss160)
print(loss161)
print("storage save to dense")
print(save80)
print(save81)
print(save160)
print(save161)
print("benefit of reorder")
print(reorder8)
print(reorder16)
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
fig, axs = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1,0.5]})
#fig=plt.subplot(2,4,gridspec_kw={'height_ratios':[2,1]})
#fig=plt.figure()

plt.subplot(2,1,1)


plt.scatter(sparsity, cnt0_8,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='number of zero vectors (V=8)(before reordering)')
plt.scatter(sparsity, cnt1_8,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='number of zero vectors (V=8)(after  reordering)')

plt.legend(loc="upper left",fontsize=24,markerscale=1.5)
# plt.ylabel("number of all-zero vector",font3, labelpad=10)
plt.ylim(0,140000)
plt.yticks(range(0, 140001, 20000))
plt.xlim(0.5,1)
plt.grid(c='grey',alpha=0.8,linestyle='--')
# plt.tick_params(labelsize=24)
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)

plt.subplot(2,1,2)
# plt.plot([0, 1], [1, 1], 'k-', color = 'black', linewidth=2, linestyle='-')
plt.scatter(sparsity, cnt0_16,s=50,c='#81d8cf',marker='o',linewidth='0.0',label='number of zero vectors (V=16)(before reordering)')
plt.scatter(sparsity, cnt1_16,s=50,c='#ffb4c8',marker='o',linewidth='0.0',label='number of zero vectors (V=16)(after  reordering)')
plt.legend(loc="upper left",fontsize=24,markerscale=1.5)
# plt.ylabel("number of all-zero vector",font3, labelpad=20)
plt.ylim(0,70000)
plt.yticks(range(0, 70001, 20000))
plt.xlim(0.5,1)
plt.xlabel("Sparsity",fontsize=24,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=20)
plt.tick_params(axis='x',labelsize=20)
plt.tight_layout()
plt.subplots_adjust(left = 0.1, right = 0.98, wspace = 0.1, hspace= 0.1)
fig.text(0.01, 0.5, 'Number of Zero Vectors', va='center', rotation='vertical', fontsize = 24)


plt.savefig('exp1-1.pdf',dpi=2000)

# plt.show()
