import matplotlib.pyplot as plt
import math

# def plot_COO(COO_format):#based on COO, plot the point of matrix
#     nnz = 0
#     row_num = len(COO_format)
#     row_ind = []
#     col_ind = []
#     for i in range(row_num):
#         # row_ind.append([i for j in range(len(COO_format[i]))])
#         for j in range(len(COO_format[i])):
#             row_ind.append(i)
#             col_ind.append(COO_format[i][j])
#     ax = plt.gca()
#     ax.xaxis.set_ticks_position('top')
#     ax.invert_yaxis()
#     plt.scatter(col_ind, row_ind, s=1, c='r',marker='s')
#     plt.show()

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
        rowInd = []
        for j in range(M):
            row_len = rowPtr[j+1] - rowPtr[j]
            for k in range(row_len):
                rowInd.append(j)
        # self.filename = path.split('/')[-1]
        self.M = M
        self.N = N
        self.nnz = nnz
        self.rowPtr = rowPtr##########array of CSR's rowPtr
        # self.rowPtr_len = rowPtr_len
        self.colInd = colInd##########array of CSR's col Index########COO's col Index
        # self.colInd_len = colInd_len
        self.rowInd = rowInd##########################################COO's row Index
    def plot_COO(self):#based on COO, plot the point of matrix
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.scatter(self.colInd, self.rowInd, s=1, c='r',marker='s')
        plt.show()
    
    def plot_point(self):#plt the point of nnz
        dist = 0
        matrix = []
        coordinate_x = []
        coordinate_y = []
        for i in range(self.M):
            row_len = self.rowPtr[i+1] - self.rowPtr[i]
            dist = self.rowPtr[i]
            row_array = []
            #print(dist)
            for j in range(row_len):
                col = self.colInd[dist+j]
                row_array.append(col)
                coordinate_x.append(col)
                coordinate_y.append(i)
                #print(col)
            #print(row_array)
            matrix.append(row_array)
        #plt.scatter(x,y,s=20,c = None,marker = 'o',cmap = none,norm = none,vmin = none,vmax = none,alpha = none,linewidths = none,verts = none,edgecolors = none,hold = none,**kwargs)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.scatter(coordinate_x, coordinate_y, s=1, c='r',marker='s')
        plt.show()
        #return matrix
    
    def colInd_list_array(self):#return matrix[ [], [], [] ] 
        ROW = self.M
        COL_IND = self.colInd
        ROW_PTR = self.rowPtr
        # origin_matrix = [[0 for i in range(self.N)] for j in range(self.M)]
        colInd_list_array = []
        for i in range(ROW):
            this_row_colind = []
            for j in COL_IND[ROW_PTR[i]:ROW_PTR[i+1]]:
                # origin_matrix[i][j] = 1
                this_row_colind.append(j)
            colInd_list_array.append(this_row_colind)
        return colInd_list_array
    
    def origin_matrix(self):#return matrix[ [], [], [] ] 
        origin_matrix = [[0 for i in range(self.N)] for j in range(self.M)]
        # array = []
        for i in range(ROW):
            this_row_colind = []
            for j in COL_IND[ROW_PTR[i]:ROW_PTR[i+1]]:
                origin_matrix[i][j] = 1
                this_row_colind.append(j)
            # array.append(this_row_colind)
    
        return origin_matrix
    """
    def return_null_row(self):##return the list of index of null_row, and its number
        smtx = self.sp_matrix_array()
        rowID = []
        row_count = 0
        for i in range(self.M):
            if smtx[i] == []:
                rowID.append(i)
                row_count += 1
        return rowID, row_count
    """
    def sparsity(self):#the whole matrix's sparsity
        sparsity_ratio = 1.0 - self.nnz / self.M / self.N 
        # print(sparsity_ratio)
        return sparsity_ratio
    """
    def row_sparsity_array(self):#list of each row's sparsity
        row_sp = []
        for i in range(self.M):
            row_len = self.rowPtr[i+1] - self.rowPtr[i]
            sparity = row_len / self.N
            #row_sp.append(sparity)
            row_sp.append(row_len)
        return row_sp
    """
class CVSE:
    def __init__(self, ROW, COL, NNZ, colInd_list_array, vec_len):

        self.M = ROW#number of rows
        self.K = COL#number of cols
        self.nnz = NNZ#number of non-zero element
        self.vec_len = vec_len#vector length
        self.num_vec_row = int(self.M / vec_len)#number of vector-row
        
        num_vec = 0
        vec_row_ptr = [0]
        vec_colInd_list_array = []#format:[[],[],...,[],[]]

        for i in range(self.num_vec_row):
            vec_colInd = []
            for j in range(self.vec_len):
                vec_colInd = list(set(vec_colInd + colInd_list_array[self.vec_len*i+j]))###this vec_row's colInd
            vec_colInd_list_array.append(vec_colInd)
            num_vec = num_vec + len(vec_colInd)
            vec_row_ptr.append(num_vec)
        self.vec_colInd_list_array = vec_colInd_list_array#format:[[],[],...,[],[]]
        self.num_vec = num_vec#number of non-zero vector
        self.vec_row_ptr = vec_row_ptr#ptr of vec_row

        vec_colInd_list = []
        for i in range(self.num_vec_row):
            for j in range(len(self.vec_colInd_list_array[i])):
                vec_colInd_list.append(self.vec_colInd_list_array[i][j])
        self.vec_colInd_list = vec_colInd_list

    def CVSE_out(self, new_dir):
        # new_path = dir + self.filename
        # print(new_dir)
        with open(new_dir, "w") as f:
            f.write(str(self.num_vec_row))
            f.write(", ")
            f.write(str(self.K))
            f.write(", ")
            f.write(str(self.nnz))
            f.write("\n")
            for i in range(len(self.vec_row_ptr)):
                f.write(str(self.vec_row_ptr[i]))
                if(i == len(self.vec_row_ptr)-1):
                    break
                f.write(" ")
            f.write("\n")
            for j in range(self.num_vec):
                # if(self.vec_colInd_list[j] >= self.K):
                #     print("\n\n\n\n\n\n\nerror\n\n\n\n\n\n")
                #     break
                f.write(str(self.vec_colInd_list[j]))
                if(j == self.num_vec - 1 ):
                    break
                f.write(" ")

def generate_CVSE_file(src_dir, dest_dir, vec_len):
        src_file_paths = src_dir + "transformer_matrices.txt"
        src_paths = []
        dest_paths = []
        with open(src_file_paths) as f:
            for i in range(2716):
                filename = f.readline().split()[0]
                full_src_dir = src_dir + filename
                full_dest_dir = dest_dir + filename
                src_paths.append(full_src_dir)
                dest_paths.append(full_dest_dir)
        # print(paths)
        # file_num = len(src_paths)
        for i in range(2716):
            matrix0 = sparse_matrix(src_paths[i])
            ROW = matrix0.M
            COL = matrix0.N
            NNZ = matrix0.nnz
            colInd_list_array = matrix0.colInd_list_array()
            
            print(src_paths[i])
            matrix1 = CVSE(ROW, COL, NNZ, colInd_list_array, vec_len)
            if(matrix1.num_vec_row == len(matrix1.vec_colInd_list_array)):
                if(matrix1.num_vec_row + 1 == len(matrix1.vec_row_ptr)):
                    if(matrix1.num_vec == matrix1.vec_row_ptr[-1]):
                        if(matrix1.num_vec == len(matrix1.vec_colInd_list)):
                            matrix1.CVSE_out(dest_paths[i])
                            print("No.", i, "  OK")
                            print(dest_paths[i])
                        else:
                            print("NO.", i, "  format error")
                            break
                    else:
                        print("NO.", i, "  format error")
                        break
                else:
                    print("NO.", i, "  format error")
                    break
            else:
                print("NO.", i, "  format error")
                break
        print("format transformation finish")
