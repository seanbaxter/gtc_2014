// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

#include "common.cuh"

// Search from needles A into haystack B. Returns lower- or upper-bound indices
// for all A needles.
template<int NT, int VT, typename T>
__global__ void KernelSortedSearchSimple(const T* a_global, int aCount,
	const T* b_global, int bCount, int* indices_global, SearchBounds bounds) {

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
		diag, bounds);

	// Sequentially merge into register starting from the partition.
	int a = mp;
	int b = aCount + diag - a;
	int aStart = a;

	int indices[VT];
	int decisions = 0;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;
		if(b >= NV) p = true;
		else if(a >= aCount) p = false;
		else p = !(data_shared[b] < data_shared[a]);
		
		if(p) {
			// aKey is smaller than bKey. Save bKey's index as the result of 
			// the search and advance to the next needle A.
			indices[i] = b - aCount;
			decisions |= 1<< i;
			++a;
		} else {
			// bKey is smaller than aKey. Advance to the next b.
			++b;
		}
	}
	__syncthreads();

	// Compact the indices to shared memory.
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		if((1<< i) & decisions)
			data_shared[aStart++] = indices[i];
	__syncthreads();

	// Store all aCount indices to global memory.
	for(int i = tid; i < aCount; i += NT)
		indices_global[i] = data_shared[i];
}


int main(int argc, char** argv) {

	const int NT = 128;
	const int VT = 7;
	const int NV = NT * VT;

	int aCount = NV / 7;
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

	int* indices_global;
	cudaMalloc2(&indices_global, aCount);

	KernelSortedSearchSimple<NT, VT><<<1, NT>>>(a_global, aCount, b_global, 
		bCount, indices_global, SearchBoundsLower);

	std::vector<int> indicesHost(aCount);
	copyDtoH(&indicesHost[0], indices_global, aCount);

	cudaFree(a_global);
	cudaFree(b_global);
	cudaFree(indices_global);

	for(int a = 0; a < aCount; ++a) {
		printf("Key %3d  index %3d", aHost[a], indicesHost[a]);

		// Print all the keys behind it.
		int begin = indicesHost[a];
		int end = (a + 1 < aCount) ? indicesHost[a + 1] : bCount;
		int count = end - begin;

		for(int i = 0; i < count; ++i) {
			if(0 == (i % 5)) printf("\n\t%3d: ", begin + i);
			printf("%3d  ", bHost[begin + i]);
		}
		printf("\n\n");
	}
	return 0;
}