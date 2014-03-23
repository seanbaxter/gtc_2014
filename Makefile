
all: reduce scan merge sortedsearch lbs

reduce: reduce.cu common.cuh
	nvcc -arch sm_20 -o reduce reduce.cu

scan: scan.cu common.cuh
	nvcc -arch sm_20 -o scan scan.cu

merge: merge.cu common.cuh
	nvcc -arch sm_20 -o merge merge.cu

sortedsearch: sortedsearch.cu common.cuh
	nvcc -arch sm_20 -o sortedsearch sortedsearch.cu

lbs: lbs.cu common.cuh
	nvcc -arch sm_20 -o lbs lbs.cu
