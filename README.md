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
IF you don't want to wait, you can download the data(dlmc-v8, dlmc-v16, dataset-v8, dataset-v16) from https://github.com/xuezy-mmi/dlmc.git.
## TAUS
```shell
cd TAUS
```
need to edit the MakeFile, change the SM_ARCH, NVCC direction.
### Build
```shell
make taus_spmm_test
```
### Run
```shell
bash ./run.sh
```
## Baseline
```shell
cd vectorsparse
```
need to edit the MakeFile, change the SM_ARCH, NVCC direction and your Sputnik direction.
### Build
```shell
make spmm_test
```
### Run
```shell
bash ./run.sh
```
