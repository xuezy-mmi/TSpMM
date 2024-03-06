# Import
import os
import ctypes
from pathlib import Path
from patoh_data import PatohData
import patoh_sugparam_enum as ps_enum
from patoh_initialize_parameters import PatohInitializeParameters


lib_path: Path = Path(os.path.join(os.getcwd(), "linux", "libpatoh.so"))

xpins_1 = [0, 5, 7, 11, 13, 15, 19, 21, 25, 27, 29, 31]
pins_1 = [2, 3, 5, 6, 9, 0, 1, 0, 1, 2, 3, 1, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 10, 11, 8, 11, 8, 10, 2, 5]
cwghts_1 = [1] * 12
nwghts_1 = [1] * 11

xpins_2 = [0, 2, 6, 9, 12]
pins_2 = [0, 2, 0, 1, 3, 4, 3, 4, 6, 2, 5, 6]
cwghts_2 = [1, 2, 3, 4, 5, 6, 7]
nwghts_2 = [11, 22, 33, 44]

# patoh_data: PatohData = PatohData(number_of_nodes=12, number_of_hyperedges=11,
#                                   node_weight_list=cwghts_1, hyperedge_weight_list=nwghts_1,
#                                   xpins=xpins_1, pins=pins_1)

patoh_data: PatohData = PatohData(number_of_nodes=7, number_of_hyperedges=4,
                                  node_weight_list=cwghts_2, hyperedge_weight_list=nwghts_2,
                                  xpins=xpins_2, pins=pins_2)

clib = ctypes.cdll.LoadLibrary(str(lib_path))

PATOH_InitializeParameters = clib.Patoh_Initialize_Parameters
PATOH_InitializeParameters.argtypes = (ctypes.POINTER(PatohInitializeParameters), ctypes.c_int, ctypes.c_int)
PATOH_Alloc = clib.Patoh_Alloc
PATOH_Alloc.argtypes = (ctypes.POINTER(PatohInitializeParameters), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
PATOH_Part = clib.Patoh_Part
PATOH_Part.argtypes = (ctypes.POINTER(PatohInitializeParameters), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
PATOH_Free = clib.Patoh_Free

# PATOH_InitializeParameters
ok = PATOH_InitializeParameters(patoh_data.parameters_ref(), 2, ps_enum.PatohSugparamEnum.PATOH_SUGPARAM_QUALITY.value)
print(f"PATOH_InitializeParameters: {ok}")

# PATOH_Alloc
ok = PATOH_Alloc(patoh_data.parameters_ref(), patoh_data.c, patoh_data.n, patoh_data.nconst,
                 patoh_data.cwghts_ctypes(), patoh_data.nwghts_ctypes(),
                 patoh_data.xpins_ctypes(), patoh_data.pins_ctypes())
print(f"PATOH_Alloc: {ok}")

# PATOH_Part
ok = PATOH_Part(patoh_data.parameters_ref(), patoh_data.c, patoh_data.n, patoh_data.nconst, patoh_data.useFixCells,
                patoh_data.cwghts_ctypes(), patoh_data.nwghts_ctypes(),
                patoh_data.xpins_ctypes(), patoh_data.pins_ctypes(),
                patoh_data.targetweights_ctypes(), patoh_data.partvec_ctypes(), patoh_data.partweights_ctypes(), patoh_data.cut_addr())
print(f"PATOH_Part: {ok}")

print(f"cut: {patoh_data.cut}")
print(f"partvec: {patoh_data.partvec()}")
print(f"partweights: {patoh_data.partweights()}")

# PATOH_Free
ok = PATOH_Free()
print(f"PATOH_Free: {ok}")

del patoh_data
