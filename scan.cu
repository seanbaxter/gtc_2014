// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

// Reduce example. This data decomposition assigns VT consecutive values to each 
// thread. Choose an odd grain-size to avoid bank conflicts when transposing 
// data through shared memory.

// Use this same thread-order data decomposition for radix sort.

#include "common.cuh"

template<int NT, int VT, typename T>
__global__ void KernelScanSimple(const T* data_global, T* scan_global) {
	const int NV = NT * VT;
	typedef CTAScan<NT, T> Scan;

	__shared__ union Shared {
		T data[NV];
		typename Scan::Storage scan;
	} shared;

	int tid = threadIdx.x;

	// Load VT elements per thread in strided order.
	// Pipelining loads from DRAM helps achieve peak bandwidth.
	T x[VT];
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		x[i] = data_global[NT * i + tid];

	// Move the data through shared memory to transpose from strided order to
	// thread order.
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		shared.data[NT * i + tid] = x[i];
	__syncthreads();

	#pragma unroll
	for(int i = 0; i < VT; ++i)
		x[i] = shared.data[VT * tid + i];
	__syncthreads();

	// Reduce the data within the thread.
	T sum = x[0];
	#pragma unroll
	for(int i = 1; i < VT; ++i)
		sum += x[i];

	// Find the exclusive scan of the intra-thread reductions across the CTA.
	T scan = Scan::Scan(tid, sum, shared.scan);

	// Incrementally add the thread-order values back into the exc scan.
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		shared.data[VT * tid + i] = scan;
		scan += x[i];
	}
	__syncthreads();

	// Read the scan values out of shared memory in strided order and store
	// to DRAM.
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		scan_global[NT * i + tid] = shared.data[NT * i + tid];
}

int main(int argc, char** argv) {

	const int NT = 128;
	const int VT = 7;
	const int NV = NT * VT;

	typedef int T;
	std::vector<T> hostData(NV, (T)1);

	T* data_global;
	cudaMalloc2(&data_global, hostData);

	KernelScanSimple<NT, VT><<<1, NT>>>(data_global, data_global);

	copyDtoH(&hostData[0], data_global, NV);

	cudaFree(data_global);

	printf("Scan of %dx%d block: \n", NT, VT);
	for(int tid = 0; tid < NT; ++tid) {
		printf("%3d: \t", tid);
		for(int i = 0; i < VT; ++i)
			printf("%4d ", hostData[VT * tid + i]);
		printf("\n");
	}

	return 0;
}