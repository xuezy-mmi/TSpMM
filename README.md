# TSpMM
poroject of TAUS

# To be continued


## Env
```shell
g++-7.5.0
cuda-11.8
sputnik
```

### Get Dataset
```shell
wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz
tar -xvf dlmc.tar.gz
rm dlmc.tar.gz
```
### Produce New Dataset
```shell
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
### Build and Run
You need to change the NVCC direction in "Makefile_xxx".
We have 4 shell script files, aimed to run on different GPU architectures.
xxx depends on your GPU architexture. xxx includes volta, turing, ampere, ada.
```shell
bash ./run_xxx.sh
```
After programming executing, some csv files will be generated in ./data/. You need to cpoy them to ../plt/ for plot figures.
```shell
cp ./data/* ../plt/
```

## Baseline
```shell
cd vectorsparse
```
need to edit "the MakeFile_xxx", change the NVCC direction and your Sputnik direction.
### Build and Run
```shell
bash ./run_baseline_xxx.sh
```

