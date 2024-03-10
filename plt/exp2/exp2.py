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

csvfile0  = pd.read_csv('../turing_taus_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile1  = pd.read_csv('../turing_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile2  = pd.read_csv('../turing_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile3  = pd.read_csv('../ampere_taus_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile4  = pd.read_csv('../ampere_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile5  = pd.read_csv('../ampere_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile6  = pd.read_csv('../ada_taus_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile7  = pd.read_csv('../ada_vectorsparse_8_v0.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])
csvfile8  = pd.read_csv('../ada_vectorsparse.csv',usecols=[0,1,2,3,4],names=["nnz","M","K","N","perf"])

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
perf0 = []
perf1 = []
perf2 = []
perf3 = []
perf4 = []
perf5 = []
perf6 = []
perf7 = []
perf8 = []
type0 = []
type1 = []
type2 = []
type3 = []
type4 = []
type5 = []
type6 = []
type7 = []
type8 = []

filenum = len(csvfile1.nnz)
# print(filenum)
for i in range (filenum):
    if(int(csvfile1.nnz[i])!=0):
        perf0.append(float(csvfile0.perf[i]))
        type0.append('TAUS-V8 Turing')
        perf1.append(float(csvfile1.perf[i]))
        type1.append('vectorSparse+RM Turing')
        perf2.append(float(csvfile2.perf[i]))
        type2.append('vectorSparse Turing')
        perf3.append(float(csvfile3.perf[i]))
        type3.append('TAUS-V8 Ampere')
        perf4.append(float(csvfile4.perf[i]))
        type4.append('vectorSparse+RM Ampere')
        perf5.append(float(csvfile5.perf[i]))
        type5.append('vectorSparse Ampere')
        perf6.append(float(csvfile6.perf[i]))
        type6.append('TAUS-V8 Ada')
        perf7.append(float(csvfile7.perf[i]))
        type7.append('vectorSparse+RM Ada')
        perf8.append(float(csvfile8.perf[i]))
        type8.append('vectorSparse Ada')

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
df_t8_turing = pd.DataFrame({
    'sparsity': sparsity0,
    'perf': perf0,
    'lib': type0
})
df_v8_turing = pd.DataFrame({
    'sparsity': sparsity1,
    'perf': perf1,
    'lib': type1
})
df_vs_turing = pd.DataFrame({
    'sparsity': sparsity2,
    'perf': perf2,
    'lib': type2
})
df_t8_ampere = pd.DataFrame({
    'sparsity': sparsity3,
    'perf': perf3,
    'lib': type3
})
df_v8_ampere = pd.DataFrame({
    'sparsity': sparsity4,
    'perf': perf4,
    'lib': type4
})
df_vs_ampere = pd.DataFrame({
    'sparsity': sparsity5,
    'perf': perf5,
    'lib': type5
})
df_t8_ada = pd.DataFrame({
    'sparsity': sparsity6,
    'perf': perf6,
    'lib': type6
})
df_v8_ada = pd.DataFrame({
    'sparsity': sparsity7,
    'perf': perf7,
    'lib': type7
})
df_vs_ada = pd.DataFrame({
    'sparsity': sparsity8,
    'perf': perf8,
    'lib': type8
})

t8turing_mean = df_t8_turing.groupby('sparsity')['perf'].mean()
total_t8turing_mean = np.mean(t8turing_mean)
v8turing_mean = df_v8_turing.groupby('sparsity')['perf'].mean()
total_v8turing_mean = np.mean(v8turing_mean)
vsturing_mean = df_vs_turing.groupby('sparsity')['perf'].mean()
total_vsturing_mean = np.mean(vsturing_mean)
t8ampere_mean = df_t8_ampere.groupby('sparsity')['perf'].mean()
total_t8ampere_mean = np.mean(t8ampere_mean)
v8ampere_mean = df_v8_ampere.groupby('sparsity')['perf'].mean()
total_v8ampere_mean = np.mean(v8ampere_mean)
vsampere_mean = df_vs_ampere.groupby('sparsity')['perf'].mean()
total_vsampere_mean = np.mean(vsampere_mean)
t8ada_mean = df_t8_ada.groupby('sparsity')['perf'].mean()
total_t8ada_mean = np.mean(t8ada_mean)
v8ada_mean = df_v8_ada.groupby('sparsity')['perf'].mean()
total_v8ada_mean = np.mean(v8ada_mean)
vsada_mean = df_vs_ada.groupby('sparsity')['perf'].mean()
total_vsada_mean = np.mean(vsada_mean)
# for i in range(7):
#     print("Turing-V8: ", sp[i], "  ", t8turing_mean[i], v8turing_mean[i])
#     print("Ampere-V8: ", sp[i], "  ", t8ampere_mean[i], v8ampere_mean[i])
#     print("Ada-V8: ", sp[i], "  ", t8ada_mean[i], v8ada_mean[i], vsada_mean[i])
print("turing:")
print(total_t8turing_mean, total_v8turing_mean, total_vsturing_mean)
print(total_t8turing_mean/total_vsturing_mean, total_v8turing_mean/total_vsturing_mean)
print("ampere:")
print(total_t8ampere_mean, total_v8ampere_mean, total_vsampere_mean)
print(total_t8ampere_mean/total_vsampere_mean, total_v8ampere_mean/total_vsampere_mean)
print("ada:")
print(total_t8ada_mean, total_v8ada_mean, total_vsada_mean)
print(total_t8ada_mean/total_vsada_mean, total_v8ada_mean/total_vsada_mean)

x = np.arange(len(sp))  # the label locations
width = 0.3  # the width of the bars

fig,ax = plt.subplots(1, 3, figsize=(30, 6))
plt.subplot(1,3,1)
plt.bar(x - width, t8turing_mean, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='TAUS-V8')
plt.bar(x , v8turing_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+RM8')
plt.bar(x + width, vsturing_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse')
for a,b in zip(x,t8turing_mean): ##label positon
    plt.text(a-width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,v8turing_mean): ##label position
    plt.text(a,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,vsturing_mean): ##label position
    plt.text(a+width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,6000)
# plt.title('Scores by group and gender',fontsize=28)
ax[0].set_xticks(x)
ax[0].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[0].set_title('RTX 2080Ti (Turing Architecture)',fontsize=24)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

plt.subplot(1,3,2)
plt.bar(x - width, t8ampere_mean, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='TAUS-V8')
plt.bar(x , v8ampere_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+RM8')
plt.bar(x + width, vsampere_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse')
for a,b in zip(x,t8ampere_mean): ##label positon
    plt.text(a-width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,v8ampere_mean): ##label position
    plt.text(a,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,vsampere_mean): ##label position
    plt.text(a+width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)

plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,6000)
# plt.title('Scores by group and gender',fontsize=28)
ax[1].set_xticks(x)
ax[1].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[1].set_title('A100 (Ampere Architecture)',fontsize=24)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)
# bbox_to_anchor=(0.95, 0.87), 
plt.subplot(1,3,3)
plt.bar(x - width, t8ada_mean, width,color='#4ea59f',edgecolor='black',linewidth=1.5,label='TAUS-V8')
plt.bar(x , v8ada_mean, width,color='#81d8cf',edgecolor='black', linewidth=1.5,label='vectorSparse+RM8')
plt.bar(x + width, vsada_mean, width,color='#ffb4c8',edgecolor='black',linewidth=1.5,label='vectorSparse')
for a,b in zip(x,t8ada_mean): ##label positon
    plt.text(a-width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,v8ada_mean): ##label position
    plt.text(a,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,vsada_mean): ##label position
    plt.text(a+width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,14000)
# plt.title('Scores by group and gender',fontsize=28)
ax[2].set_xticks(x)
ax[2].set_xticklabels(sp,rotation=0, fontsize = 22)
ax[2].set_title('RTX 4090 (Ada Architecture)',fontsize=24)
plt.tick_params(labelsize=22)
plt.grid(c='grey',alpha=0.9,linestyle='--')
plt.legend(loc="upper right", borderaxespad=0,fontsize=22,ncol=1)

fig.tight_layout()
plt.savefig('exp2.pdf',dpi=300)
# plt.show()
