// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

// Reduce example. This data decomposition assigns VT strided values to each 
// thread.

#include "common.cuh"

template<int NT, int VT, typename T>
__global__ void KernelReduceSimple(const T* data_global, T* reduce_global) {

	typedef CTAReduce<NT, T> Reduce;
	__shared__ typename Reduce::Storage reduce_shared;

	int tid = threadIdx.x;

	// Load VT elements per thread in strided order.
	// Pipelining loads from DRAM helps achieve peak bandwidth.
	T x[VT];
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		x[i] = data_global[NT * i + tid];

	// Reduce the elements within the thread. This requires commutativity.
	T sum = x[0];
	#pragma unroll
	for(int i = 1; i < VT; ++i)
		sum += x[i];

	// Reduce the elements across the CTA.
	sum = Reduce::Reduce(tid, sum, reduce_shared);

	// Thread 0 stores the result to DRAM.
	if(!tid) 
		reduce_global[0] = sum;
}

int main(int argc, char** argv) {

	const int NT = 128;
	const int VT = 8;
	const int NV = NT * VT;

	typedef int T;
	std::vector<T> hostData(NV, (T)1);

	T* data_global;
	cudaMalloc2(&data_global, hostData);

	T* reduce_global;
	cudaMalloc2(&reduce_global, 1);

	KernelReduceSimple<NT, VT><<<1, NT>>>(data_global, reduce_global);

	T total;
	copyDtoH(&total, reduce_global, 1);

	cudaFree(reduce_global);
	cudaFree(data_global);

	printf("Reduction of %dx%d block: %d\n", NT, VT, total);

	return 0;
}