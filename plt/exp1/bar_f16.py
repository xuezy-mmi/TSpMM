import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 0:name
# 1:dasp-a100
# 2:cusparse-a100
# 3:dasp-h800
# 4:cusparse-h800


labels = ['dc2', 'scir...', 'mac_...', 'webb...', 'ASIC...', 'Full...', 'rma1...', 
          'eu-2...', 'in-2...', 'cant', 'circ...', 'cop2...', 'mc2d...', 'pdb1...', 
          'conf...','cons...', 'ship...', 'Si41...', 'mip1', 'pwtk','Ga41...']

sizeofr = 21
sizeofc = 21

content1 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content2 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content3 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content4 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content5 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]
# content6 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]


font3 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 18,}

with open('fp16_21.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for num in range(sizeofr):
        content1[num] = rows[num]

data_cu_a = [0.0 for i in range(sizeofr)]
data_cu_h = [0.0 for i in range(sizeofr)]
data_dasp_h = [0.0 for i in range(sizeofr)]
data_dasp_a = [0.0 for i in range(sizeofr)]


for num in range(sizeofr):
    data_dasp_a[num] = (float)(content1[num][1])
    data_cu_a[num] = (float)(content1[num][2])
    data_dasp_h[num] = (float)(content1[num][3])
    data_cu_h[num] = (float)(content1[num][4])

print(data_dasp_a[0], data_cu_a[0], data_dasp_h[0], data_cu_h[0])
x = np.arange(len(labels))  # the label locations
# y = np.arange(len(men_means))
width = 0.17  # the width of the bars

fig,ax = plt.subplots(figsize=(36, 5))
# plt.bar(x - width, data_tile, width,color='lightseagreen',edgecolor='black',linewidth=1.5,label='TileSpMV')
# plt.bar(x , data_cu, width,color='darkcyan',edgecolor='black', linewidth=1.5,label='cuSPARSE')
# plt.bar(x + width , data_csr5, width,color='lightsteelblue',edgecolor='black', linewidth=1.5, label='CSR5')
# plt.bar(x + (width*2), data_dasp, width,color='steelblue',edgecolor='black',linewidth=1.5, label='DASP')
plt.bar(x - width, data_cu_a, width,color='#c7dbd5',edgecolor='black',linewidth=1.5,label='cuSPARSE (v12.0) on A100')
plt.bar(x + width , data_cu_h, width,color='#f6b654',edgecolor='black', linewidth=1.5, label='cuSPARSE (v12.0) on H800')
plt.bar(x , data_dasp_a, width,color='#4ea59f',edgecolor='black', linewidth=1.5,label='DASP (this work) on A100')

plt.bar(x + (width*2), data_dasp_h, width,color='#ee6a5b',edgecolor='black',linewidth=1.5, label='DASP (this work) on H800')




for a,b in zip(x,data_dasp_a): ##控制标签位置
    plt.text(a,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,data_cu_a): ##控制标签位置
    plt.text(a-width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
# for a,b in zip(x,data_cu): ##控制标签位置
#     plt.text(a-(width*2),b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,data_dasp_h): ##控制标签位置
    # if a == 8:
    #     plt.text(a+(width*2),b-30,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
    # else:
        plt.text(a+(width*2),b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)
for a,b in zip(x,data_cu_h): ##控制标签位置
    plt.text(a+width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)




# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,650)
# plt.title('Scores by group and gender',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,rotation=0, fontsize = 22)
plt.tick_params(labelsize=22)

plt.grid(c='grey',alpha=0.8,linestyle='--')

fig.legend(bbox_to_anchor=(0.33, 0.87), loc=1, borderaxespad=0,fontsize=22,ncol=2)


fig.tight_layout()
plt.savefig('mtxbar_f16_2.pdf',dpi=300)
# plt.show()




