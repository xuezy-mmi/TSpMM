#!/usr/bin/python3
import os
import csv
import random
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import hypergraph

tmp_path = "/home/xuezeyu/dlmc/transformer/l0_regularization/0.6/body_decoder_layer_5_encdec_attention_multihead_attention_v.smtx"
transformer_matrices_data_dir = "dlmc/transformer_matrices.txt"

def bool_vec_mate(v1, v2, m):#  0 < m < 1
    l1 = len(v1)
    l2 = len(v2)
    #v1.extend(v2)
    v3 = []
    v3.extend(v1)
    for i in v2:
        if i not in v3:
            v3.append(i)
    l3 = len(v3)
    if((l3 < l1+m*l2) or (l3 < m*l1+l2)):######
        return True
    else:
        return False

def bool_vec_complementary(v1, v2, k):#k = 16 / 32
    #l1 = len(v1)
    #l2 = len(v2)
    #v1.extend(v2)
    v3 = []
    v3.extend(v1)
    for i in v2:
        if i not in v1:
            v3.append(i)
    l3 = len(v3)
    if l3 <= k:
        return True
    else:
        return False
    
def Utilization_of_TC_M_K(M, K, origin_mtx, reorder_mtx):
    # M and K is MKN of WMMA-inst
    # origin_mtx is 2D array[[]*m]
    # reorder_mtx is 2D array[[]*m]
    group = M / K
    
class sparse_matrix:
    def __init__(self, path):
        with open(path) as f:#####read from .smtx and init
            data = f.readline()
            rowPtr_s = f.readline()
            colInd_s = f.readline()
            M, N, nnz = data.split(', ')
            rowPtr = rowPtr_s.split()
            colInd = colInd_s.split()
            rowPtr_len = len(rowPtr)
            colInd_len = len(colInd)
        f.close()
        M = int(M)
        N = int(N)
        nnz = int(nnz)
        for i in range(rowPtr_len):
            rowPtr[i] = int(rowPtr[i])
        for i in range(colInd_len):
            colInd[i] = int(colInd[i])
        self.M = M
        self.N = N
        self.nnz = nnz
        self.rowPtr = rowPtr##########array of CSR's rowPtr
        self.rowPtr_len = rowPtr_len
        self.colInd = colInd##########array of CSR's colInd
        self.colInd_len = colInd_len

def return_size(mtx):
    row_num = len(mtx)
    col_num = len(mtx[0])
    return row_num, col_num

def return_zero_col_num(mtx, M=8):#default M = 8
    row_num, col_num = return_size(mtx)
    group = int(row_num/M)
    zero_col_num = [0 for k in range(group)]
    count = 0
    for i in range(group):
        #zero_col_num[i] = 0
        for j in range(col_num):
            #if(mtx[i*M][j] == mtx[i*M+1][j] == mtx[i*M+2][j] == mtx[i*M+3][j] == mtx[i*M+4][j] == mtx[i*M+5][j] == mtx[i*M+6][j] == mtx[i*M+7][j] == 0):
            flag = 1
            for ii in range(M):
                if(mtx[i*M+ii][j] != 0):
                    flag = 0
                    break
            if(flag == 1):
                zero_col_num[i] = zero_col_num[i] + 1
                count = count + 1
    return zero_col_num, count

def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)
    
def utilization_of_TC(data_dir, inst_M):
    with open("output32-1.csv","w") as csvfile:
        writer = csv.writer(csvfile)

        #先写入columns_name
        writer.writerow(["sparsity","utilization0","utilization1","cnto","cnt1"])

        file_num = count_lines(data_dir)
        for i in range(file_num):
            origin_matrix, reorder_matrix, nnz = hypergraph.Processing_Matrices_Transformer(data_dir, i, inst_M)
            row_num, col_num = return_size(origin_matrix)
            zero_col_num0, cnt0 = return_zero_col_num(origin_matrix, inst_M)
            zero_col_num1, cnt1 = return_zero_col_num(reorder_matrix, inst_M)
            if(nnz == 0):
                writer.writerow([1, 0, 0, 0, 0])
            else:
                tile_num0 = 0
                tile_num1 = 0
                #print(len(zero_col_num0), len(zero_col_num1))
                for i in range(len(zero_col_num0)):
                    none_zero_col_num =  col_num - zero_col_num0[i]
                    tile_num0 = tile_num0 + math.ceil(none_zero_col_num/16)
                utilization0 = nnz / (tile_num0 * 16 * inst_M)
                for j in range(len(zero_col_num1)):
                    none_zero_col_num =  col_num - zero_col_num1[j]
                    tile_num1 = tile_num1 + math.ceil(none_zero_col_num/16)
                utilization1 = nnz / (tile_num1 * 16 * inst_M)
                sparsity = 1-nnz/row_num/col_num
                print("sparsity\tutilization0\tutilization1\tcnt0\tcnt1")
                print("{:.4f}%".format(sparsity),"\t","{:.4f}%".format(utilization0),"\t","{:.4f}%".format(utilization1),"\t",cnt0,"\t",cnt1)

                writer.writerow([sparsity, utilization0, utilization1, cnt0, cnt1])


if __name__ == '__main__':

    utilization_of_TC(transformer_matrices_data_dir, 32)
