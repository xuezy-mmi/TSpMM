# TSpMM
poroject of TAUS

# To be continued


## Env
```shell
g++-7.5.0
cuda-11.8
sputnik
```

## Get Dataset
```shell

wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
tar -xvf dlmc.tar.gz
rm dlmc.tar.gz
cd RowMerge
bash mkdir.sh
python rowmerge.py
```
wait about 6hours
## TAUS

```shell
cd TAUS
```
need to edit the MakeFile, change the SM_ARCH, NVCC direction, 

### Build
```shell
make taus_spmm_test
```



### Run
```shell
make taus_spmm_test_run_XXX
```
XXX is the architecture of your GPU(volta, turing, ampere, ada)
