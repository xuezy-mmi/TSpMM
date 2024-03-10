import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


font3 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 18,}

csvfile0  = pd.read_csv('../volta_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile1  = pd.read_csv('../volta_vectorsparse_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile2  = pd.read_csv('../volta_vectorsparse_8_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile3  = pd.read_csv('../volta_vectorsparse_16_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile4  = pd.read_csv('../turing_taus_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile5  = pd.read_csv('../turing_taus_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile6  = pd.read_csv('../turing_taus_8_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile7  = pd.read_csv('../turing_taus_16_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile8  = pd.read_csv('../ampere_taus_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile9  = pd.read_csv('../ampere_taus_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile10 = pd.read_csv('../ampere_taus_8_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile11 = pd.read_csv('../ampere_taus_16_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile12 = pd.read_csv('../ada_taus_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile13 = pd.read_csv('../ada_taus_16_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile14 = pd.read_csv('../ada_taus_8_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile15 = pd.read_csv('../ada_taus_16_v1.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

sp = ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.98"]
sparsity0 = []
sparsity1 = []
sparsity2 = []
sparsity3 = []
sparsity4 = []
sparsity5 = []
sparsity6 = []
sparsity7 = []
sparsity8 = []
sparsity9 = []
sparsity10 = []
sparsity11 = []
sparsity12 = []
sparsity13 = []
sparsity14 = []
sparsity15 = []
perf0 = []
perf1 = []
perf2 = []
perf3 = []
perf4 = []
perf5 = []
perf6 = []
perf7 = []
perf8 = []
perf9 = []
perf10 = []
perf11 = []
perf12 = []
perf13 = []
perf14 = []
perf15 = []
type0 = []
type1 = []
type2 = []
type3 = []
type4 = []
type5 = []
type6 = []
type7 = []
type8 = []
type9 = []
type10 = []
type11 = []
type12 = []
type13 = []
type14 = []
type15 = []
filenum = len(csvfile1.nnz)
print(filenum)
for i in range (filenum):
    if(int(csvfile1.nnz[i])!=0):
        perf0.append(float(csvfile0.perf[i]))
        type0.append('TAUS-V8 Volta')
        perf1.append(float(csvfile1.perf[i]))
        type1.append('TAUS-V16 Volta')
        perf2.append(float(csvfile2.perf[i]))
        type2.append('TAUS-V8(reordering) Volta')
        perf3.append(float(csvfile3.perf[i]))
        type3.append('TAUS-V16(reordering) Volta')
        perf4.append(float(csvfile4.perf[i]))
        type4.append('TAUS-V8 Turing')
        perf5.append(float(csvfile5.perf[i]))
        type5.append('TAUS-V16 Turing')
        perf6.append(float(csvfile6.perf[i]))
        type6.append('TAUS-V8(reordering) Turing')
        perf7.append(float(csvfile7.perf[i]))
        type7.append('TAUS-V16(reordering) Turing')
        perf8.append(float(csvfile8.perf[i]))
        type8.append('TAUS-V8 Ampere')
        perf9.append(float(csvfile9.perf[i]))
        type9.append('TAUS-V16 Ampere')
        perf10.append(float(csvfile10.perf[i]))
        type10.append('TAUS-V8(reordering) Ampere')
        perf11.append(float(csvfile11.perf[i]))
        type11.append('TAUS-V16(reordering) Ampere')
        perf12.append(float(csvfile12.perf[i]))
        type12.append('TAUS-V8 Ada')
        perf13.append(float(csvfile13.perf[i]))
        type13.append('TAUS-V16 Ada')
        perf14.append(float(csvfile14.perf[i]))
        type14.append('TAUS-V8(reordering) Ada')
        perf15.append(float(csvfile15.perf[i]))
        type15.append('TAUS-V16(reordering) Ada')
for i in range(4):
    for j in range(7):
        xlabel = sp[j]
        for k in range(97):
            NO_file = i*7*97+j*97+k
            if(int(csvfile1.nnz[NO_file])!=0):
                sparsity0.append(xlabel)
                sparsity1.append(xlabel)
                sparsity2.append(xlabel)
                sparsity3.append(xlabel)
                sparsity4.append(xlabel)
                sparsity5.append(xlabel)
                sparsity6.append(xlabel)
                sparsity7.append(xlabel)
                sparsity8.append(xlabel)
                sparsity9.append(xlabel)
                sparsity10.append(xlabel)
                sparsity11.append(xlabel)
                sparsity12.append(xlabel)
                sparsity13.append(xlabel)
                sparsity14.append(xlabel)
                sparsity15.append(xlabel)

df_t80volta = pd.DataFrame({
    'sparsity': sparsity0,
    'perf': perf0,
    'lib': type0
})
df_t160volta = pd.DataFrame({
    'sparsity': sparsity1,
    'perf': perf1,
    'lib': type1
})
df_t81volta = pd.DataFrame({
    'sparsity': sparsity2,
    'perf': perf2,
    'lib': type2
})
df_t161volta = pd.DataFrame({
    'sparsity': sparsity3,
    'perf': perf3,
    'lib': type3
})
df_t80turing = pd.DataFrame({
    'sparsity': sparsity4,
    'perf': perf4,
    'lib': type4
})
df_t160turing = pd.DataFrame({
    'sparsity': sparsity5,
    'perf': perf5,
    'lib': type5
})
df_t81turing = pd.DataFrame({
    'sparsity': sparsity6,
    'perf': perf6,
    'lib': type6
})
df_t161turing = pd.DataFrame({
    'sparsity': sparsity7,
    'perf': perf7,
    'lib': type7
})
df_t80ampere = pd.DataFrame({
    'sparsity': sparsity8,
    'perf': perf8,
    'lib': type8
})
df_t160ampere = pd.DataFrame({
    'sparsity': sparsity9,
    'perf': perf9,
    'lib': type9
})
df_t81ampere = pd.DataFrame({
    'sparsity': sparsity10,
    'perf': perf10,
    'lib': type10
})
df_t161ampere = pd.DataFrame({
    'sparsity': sparsity11,
    'perf': perf11,
    'lib': type11
})
df_t80ada = pd.DataFrame({
    'sparsity': sparsity12,
    'perf': perf12,
    'lib': type12
})
df_t160ada = pd.DataFrame({
    'sparsity': sparsity13,
    'perf': perf13,
    'lib': type13
})
df_t81ada = pd.DataFrame({
    'sparsity': sparsity14,
    'perf': perf14,
    'lib': type14
})
df_t161ada = pd.DataFrame({
    'sparsity': sparsity15,
    'perf': perf15,
    'lib': type15
})
t80volta_mean = df_t80volta.groupby('sparsity')['perf'].mean()
t160volta_mean = df_t160volta.groupby('sparsity')['perf'].mean()
trade_off00 = [0.0 for i in range(7)]
t81volta_mean = df_t81volta.groupby('sparsity')['perf'].mean()
t161volta_mean = df_t161volta.groupby('sparsity')['perf'].mean()
reorder81 = [0.0 for i in range(7)]
reorder161 = [0.0 for i in range(7)]
trade_off01 = [0.0 for i in range(7)]
t80turing_mean = df_t80turing.groupby('sparsity')['perf'].mean()
t160turing_mean = df_t160turing.groupby('sparsity')['perf'].mean()
trade_off10 = [0.0 for i in range(7)]
t81turing_mean = df_t81turing.groupby('sparsity')['perf'].mean()
t161turing_mean = df_t161turing.groupby('sparsity')['perf'].mean()
reorder82 = [0.0 for i in range(7)]
reorder162 = [0.0 for i in range(7)]
trade_off11 = [0.0 for i in range(7)]
t80ampere_mean = df_t80ampere.groupby('sparsity')['perf'].mean()
t160ampere_mean = df_t160ampere.groupby('sparsity')['perf'].mean()
trade_off20 = [0.0 for i in range(7)]
t81ampere_mean = df_t81ampere.groupby('sparsity')['perf'].mean()
t161ampere_mean = df_t161ampere.groupby('sparsity')['perf'].mean()
trade_off21 = [0.0 for i in range(7)]
reorder83 = [0.0 for i in range(7)]
reorder163 = [0.0 for i in range(7)]
t80ada_mean = df_t80ada.groupby('sparsity')['perf'].mean()
t160ada_mean = df_t160ada.groupby('sparsity')['perf'].mean()
trade_off30 = [0.0 for i in range(7)]
t81ada_mean = df_t81ada.groupby('sparsity')['perf'].mean()
t161ada_mean = df_t161ada.groupby('sparsity')['perf'].mean()
trade_off31 = [0.0 for i in range(7)]
reorder84 = [0.0 for i in range(7)]
reorder164 = [0.0 for i in range(7)]

for i in range(7):
    # print("Ada-V8: ", sp[i], "  ", t80ada_mean[i], t160ada_mean[i])
    trade_off00[i] = t80volta_mean[i]/t160volta_mean[i]
    trade_off01[i] = t81volta_mean[i]/t161volta_mean[i]
    trade_off10[i] = t80turing_mean[i]/t160turing_mean[i]
    trade_off11[i] = t81turing_mean[i]/t161turing_mean[i]
    trade_off20[i] = t80ampere_mean[i]/t160ampere_mean[i]
    trade_off21[i] = t81ampere_mean[i]/t161ampere_mean[i]
    trade_off30[i] = t80ada_mean[i]/t160ada_mean[i]
    trade_off31[i] = t81ada_mean[i]/t161ada_mean[i]
    
    reorder81[i] = t81volta_mean[i]/t80volta_mean[i]
    reorder161[i] = t161volta_mean[i]/t160volta_mean[i]
    reorder82[i] = t81turing_mean[i]/t80turing_mean[i]
    reorder162[i] = t161turing_mean[i]/t160turing_mean[i]
    reorder83[i] = t81ampere_mean[i]/t80ampere_mean[i]
    reorder163[i] = t161ampere_mean[i]/t160ampere_mean[i]
    reorder84[i] = t81ada_mean[i]/t80ada_mean[i]
    reorder164[i] = t161ada_mean[i]/t160ada_mean[i]

reorder8_volta = sum(t81volta_mean[4:7]) / sum(t80volta_mean[4:7])
reorder16_volta = sum(t161volta_mean[4:7]) / sum(t160volta_mean[4:7])
reorder8_turing = sum(t81turing_mean[4:7]) / sum(t80turing_mean[4:7])
reorder16_turing = sum(t161turing_mean[4:7]) / sum(t160turing_mean[4:7])
reorder8_ampere = sum(t81ampere_mean[4:7]) / sum(t80ampere_mean[4:7])
reorder16_ampere = sum(t161ampere_mean[4:7]) / sum(t160ampere_mean[4:7])
reorder8_ada = sum(t81ada_mean[4:7]) / sum(t80ada_mean[4:7])
reorder16_ada = sum(t161ada_mean[4:7]) / sum(t160ada_mean[4:7])
print("volta  reorder8  (>90%): ",reorder8_volta)
print("volta  reorder16 (>90%): ",reorder16_volta)
print("turing reorder8  (>90%): ",reorder8_turing)
print("turing reorder16 (>90%): ",reorder16_turing)
print("ampere reorder8  (>90%): ",reorder8_ampere)
print("ampere reorder16 (>90%): ",reorder16_ampere)
print("ada    reorder8  (>90%): ",reorder8_ada)
print("ada    reorder16 (>90%): ",reorder16_ada)
for i in range(7):
    print(sp[i], "  trade-off")
    print("volta   8-16: ", trade_off00[i])
    print("reorder 8-16: ", trade_off01[i])
    print("turing  8-16: ", trade_off10[i])
    print("reorder 8-16: ", trade_off11[i])
    print("ampere  8-16: ", trade_off20[i])
    print("reorder 8-16: ", trade_off21[i])
    print("ada     8-16: ", trade_off30[i])
    print("reorder 8-16: ", trade_off31[i])
# for i in range(7):
#     print(sp[i]," reorder ")
#     print("volta v8   ", reorder81[i])
#     print("volta v16  ", reorder161[i])
#     print("turing v8  ", reorder82[i])
#     print("turing v16 ", reorder162[i])
#     print("ampere v8  ", reorder83[i])
#     print("ampere v16 ", reorder163[i])
#     print("ada v8     ", reorder84[i])
#     print("ada v16    ", reorder164[i])
    
x = np.arange(len(sp))  # the label locations
# y = np.arange(len(men_means))
width = 0.2  # the width of the bars

fig,ax = plt.subplots(2, 2, figsize=(30, 12))
# fig, axs = plt.subplots(2, 2, figsize=(20, 10))
plt.subplot(2,2,1)
plt.bar(x - width, t80volta_mean, width,color='#81d8cf',edgecolor='black',linewidth=1.5,label='vectorSparse+RM8')
plt.bar(x , t160volta_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='vectorSparse+RM16')
plt.bar(x + width , t81volta_mean, width,color='#ffb4c8',edgecolor='black', linewidth=1.5, label='vectorSparse+RM8(reordering)')
plt.bar(x + (width*2), t161volta_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5, label='vectorSparse+RM16(reordering)')
for a,b in zip(x,t80volta_mean): ##label position
    plt.text(a-width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t160volta_mean): ##label position
    plt.text(a,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t81volta_mean): ##label position
    plt.text(a+width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t161volta_mean): ##label position
    plt.text(a+(width*2),b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,7000)
# plt.title('Scores by group and gender',fontsize=28)
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[0,0].set_title('V100 (Volta Architecture)',fontsize=24)
plt.xlabel('Sparsity',fontsize=18)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

ax2 = ax[0,0].twinx()
plt.plot(x-width*0.5, trade_off00, marker='o', linestyle='-', color = 'blue', label = 'TAUS-V8 / TAUS-V16')
plt.plot(x+width*1.5, trade_off01, marker='o', linestyle='-', color = 'red', label = 'TAUS-V8(reordering) / TAUS-V16(reordering)')
ax2.set_ylabel('V8 / V16 (ratio)', fontsize=22)
ax2.set_ylim(0, 1.4)
plt.tick_params(labelsize=18)

# plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

plt.subplot(2,2,2)
plt.bar(x - width, t80turing_mean, width,color='#81d8cf',edgecolor='black',linewidth=1.5,label='TAUS-V8')
plt.bar(x , t160turing_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='TAUS-V16')
plt.bar(x + width , t81turing_mean, width,color='#ffb4c8',edgecolor='black', linewidth=1.5, label='TAUS-V8(reordering)')
plt.bar(x + (width*2), t161turing_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5, label='TAUS-V16(reordering)')
for a,b in zip(x,t80turing_mean): ##label position
    plt.text(a-width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t160turing_mean): ##label position
    plt.text(a,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t81turing_mean): ##label position
    plt.text(a+width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t161turing_mean): ##label position
    plt.text(a+(width*2),b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,6000)
# plt.title('Scores by group and gender',fontsize=28)
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[0,1].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
plt.xlabel('Sparsity',fontsize=18)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

ax2 = ax[0,1].twinx()
plt.plot(x-width*0.5, trade_off10, marker='o', linestyle='-', color = 'blue', label = 'TAUS-V8 / TAUS-V16')
plt.plot(x+width*1.5, trade_off11, marker='o', linestyle='-', color = 'red', label = 'TAUS-V8(reordering) / TAUS-V16(reordering)')
ax2.set_ylabel('V8 / V16 (ratio)', fontsize=22)
ax2.set_ylim(0, 1.4)
plt.tick_params(labelsize=18)

# bbox_to_anchor=(0.95, 0.87), 
plt.subplot(2,2,3)
plt.bar(x - width, t80ampere_mean, width,color='#81d8cf',edgecolor='black',linewidth=1.5,label='TAUS-V8')
plt.bar(x , t160ampere_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='TAUS-V16')
plt.bar(x + width , t81ampere_mean, width,color='#ffb4c8',edgecolor='black', linewidth=1.5, label='TAUS-V8(reordering)')
plt.bar(x + (width*2), t161ampere_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5, label='TAUS-V16(reordering)')
for a,b in zip(x,t80ampere_mean): ##label position
    plt.text(a-width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t160ampere_mean): ##label position
    plt.text(a,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t81ampere_mean): ##label position
    plt.text(a+width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t161ampere_mean): ##label position
    plt.text(a+(width*2),b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,6000)
# plt.title('Scores by group and gender',fontsize=28)
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[1,0].set_title('A100 (Ampere Architecture)',fontsize=24)
plt.xlabel('Sparsity',fontsize=18)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
ax2 = ax[1,0].twinx()
plt.plot(x-width*0.5, trade_off20, marker='o', linestyle='-', color = 'blue', label = 'TAUS-V8 / TAUS-V16')
plt.plot(x+width*1.5, trade_off21, marker='o', linestyle='-', color = 'red', label = 'TAUS-V8(reordering) / TAUS-V16(reordering)')
ax2.set_ylabel('V8 / V16 (ratio)', fontsize=22)
ax2.set_ylim(0, 1.4)
plt.tick_params(labelsize=18)

plt.subplot(2,2,4)
plt.bar(x - width, t80ada_mean, width,color='#81d8cf',edgecolor='black',linewidth=1.5,label='TAUS-V8')
plt.bar(x , t160ada_mean, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='TAUS-V16')
plt.bar(x + width , t81ada_mean, width,color='#ffb4c8',edgecolor='black', linewidth=1.5, label='TAUS-V8(reordering)')
plt.bar(x + (width*2), t161ada_mean, width,color='#ee6a5b',edgecolor='black',linewidth=1.5, label='TAUS-V16(reordering)')
for a,b in zip(x,t80ada_mean): ##label position
    plt.text(a-width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t160ada_mean): ##label position
    plt.text(a,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t81ada_mean): ##label position
    plt.text(a+width,b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,t161ada_mean): ##label position
    plt.text(a+(width*2),b+2,'%.1f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,14000)
# plt.title('Scores by group and gender',fontsize=28)
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[1,1].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
plt.xlabel('Sparsity',fontsize=18)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

ax2 = ax[1,1].twinx()
plt.plot(x-width*0.5, trade_off30, marker='o', linestyle='-', color = 'blue', label = 'TAUS-V8 / TAUS-V16')
plt.plot(x+width*1.5, trade_off31, marker='o', linestyle='-', color = 'red', label = 'TAUS-V8(reordering) / TAUS-V16(reordering)')
ax2.set_ylabel('V8 / V16 (ratio)', fontsize=22)
ax2.set_ylim(0, 1.4)
plt.tick_params(labelsize=18)

fig.tight_layout()
plt.savefig('exp3.pdf',dpi=300)
# plt.show()
