// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

#include "common.cuh"

template<int NT, int VT, typename T>
__global__ void KernelMergeSimple(const T* a_global, int aCount,
	const T* b_global, int bCount, T* merged_global) {

	const int NV = NT * VT;
	__shared__ T data_shared[NT * VT + 1];

	int tid = threadIdx.x;

	// Load aCount elements and NV - aCount elements from b_global.
	b_global -= aCount;
	
	T x[VT];
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < aCount) x[i] = a_global[index];
		else x[i] = b_global[index];
	}
	
	// Store all elements to shared memory.
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		data_shared[NT * i + tid] = x[i];
	__syncthreads();

	// Each thread searches for its Merge Path partition.
	int diag = VT * tid;
	int mp = MergePath(data_shared, aCount, data_shared + aCount, bCount,
		diag);

	// Sequentially merge into register starting from the partition.
	int a = mp;
	int b = aCount + diag - a;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;
		if(b >= NV) p = true;
		else if(a >= aCount) p = false;
		else p = !(data_shared[b] < data_shared[a]);
		
		x[i] = p ? data_shared[a++] : data_shared[b++];
	}
	__syncthreads();

	// The merged data is now in thread order in register. Transpose through
	// shared memory and store to DRAM.
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		data_shared[VT * tid + i] = x[i];
	__syncthreads();

	#pragma unroll
	for(int i = 0; i < VT; ++i)
		merged_global[NT * i + tid] = data_shared[NT * i + tid];
}

int main(int argc, char** argv) {

	const int NT = 128;
	const int VT = 7;
	const int NV = NT * VT;

	int aCount = NV / 2;
	int bCount = NV - aCount;

	// Generate random sorted arrays to merge.
	std::vector<int> aHost(aCount), bHost(bCount);
	for(int i = 0; i < aCount; ++i)
		aHost[i] = rand() % 100;
	for(int i = 0; i < bCount; ++i)
		bHost[i] = rand() % 100;

	std::sort(aHost.begin(), aHost.end());
	std::sort(bHost.begin(), bHost.end());

	int* a_global, *b_global;
	cudaMalloc2(&a_global, aHost);
	cudaMalloc2(&b_global, bHost);

	int* merged_global;
	cudaMalloc2(&merged_global, NV);

	KernelMergeSimple<NT, VT><<<1, NT>>>(a_global, aCount, b_global, bCount,
		merged_global);

	std::vector<int> mergedHost(NV);
	cudaDtoH(&mergedHost[0], merged_global, NV);

	cudaFree(a_global);
	cudaFree(b_global);
	cudaFree(merged_global);

	for(int tid = 0; tid < NT; ++tid) {
		printf("%3d: \t", tid);

		for(int i = 0; i < VT; ++i)
			printf("%3d ", mergedHost[VT * tid + i]);
		printf("\n");
	}
	return 0;
}
