// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

#include "common.cuh"

// Search from needles A into haystack B. Returns lower- or upper-bound indices
// for all A needles.
template<int NT, int VT>
__global__ void KernelLBSSimple(int aCount, const int* b_global, int bCount,
	int* indices_global) {

	__shared__ int data_shared[NT * VT];

	int tid = threadIdx.x;

	// Load bCount elements from B into data_shared.
	int x[VT];
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < bCount) x[i] = b_global[index];
	}
	
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		data_shared[NT * i + tid] = x[i];
	__syncthreads();

	// Each thread searches for its Merge Path partition.
	int diag = VT * tid;
	int begin = max(0, diag - bCount);
	int end = min(diag, aCount);

	while(begin < end) {
		int mid = (begin + end)>> 1;
		int aKey = mid;
		int bKey = data_shared[diag - 1 - mid];
		bool pred = aKey < bKey;
		if(pred) begin = mid + 1;
		else end = mid;
	}
	int mp = begin;

	// Sequentially search, comparing indices a to elements data_shared[b].
	// Store indices for A in the right-side of the shared memory array.
	// This lets us complete the search in just a single pass, rather than 
	// the search and compact passes of the generalized vectorized sorted
	// search function.
	int a = mp;
	int b = diag - a;
	
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;
		if(b >= bCount) p = true;
		else if(a >= aCount) p = false;
		else p = a < data_shared[b];
		
		if(p)
			// If a < data_shared[b], advance A and store the index b - 1.
			data_shared[bCount + a++] = b - 1;
		else
			// Just advance b.
			++b;
	}
	__syncthreads();

	// Store all indices to global memory.
	for(int i = tid; i < aCount; i += NT)
		indices_global[i] = data_shared[bCount + i];
}

// Generate a CSR array by recursively bisecting and setting a random value.
void GenCSRArrayRecurse(int* csr, int begin, int end, int left, int right) {
	int mid = (begin + end) / 2;
	int val;
	if(right - left > 10)
		val = (left + right - 10) / 2 + (rand() % 10);
	else
		val = left + (rand() % (right - left + 1));

	csr[mid] = val;
	if(mid > begin) GenCSRArrayRecurse(csr, begin, mid, left, val);
	if(mid + 1 < end) GenCSRArrayRecurse(csr, mid + 1, end, val, right);
}

void GenCSRArray(int aCount, int bCount, std::vector<int>& csr) {
	csr.resize(bCount + 1);
	csr[0] = 0;
	csr[bCount] = aCount;
	GenCSRArrayRecurse(&csr[0], 1, bCount, 0, aCount);
	
	csr.resize(bCount);
}

int main(int argc, char** argv) {

	const int NT = 128;
	const int VT = 7;
	const int NV = NT * VT;

	int bCount = NV / 7;
	int aCount = NV - bCount;

	// Generate the CSR array.
	std::vector<int> csrHost;
	GenCSRArray(aCount, bCount, csrHost);

	// Allocate GPU device memory.
	int* a_global, *b_global;
	cudaMalloc2(&a_global, aCount);
	cudaMalloc2(&b_global, csrHost);

	// Run the CTA-wide LBS.
	KernelLBSSimple<NT, VT><<<1, NT>>>(aCount, b_global, bCount, a_global);

	// Retrieve the COO array.
	std::vector<int> cooHost(aCount);
	copyDtoH(&cooHost[0], a_global, aCount);

	cudaFree(a_global);
	cudaFree(b_global);

	// Print both arrays.
	for(int b = 0; b < bCount; ++b) {
		printf("%3d (%3d):", b, csrHost[b]);

		int begin = csrHost[b];
		int end = (b + 1 < bCount) ? csrHost[b + 1] : aCount;
		int count = end - begin;
		
		for(int i = 0; i < count; ++i) {
			if(0 == (i % 5)) printf("\n  %3d: \t", begin + i);
			printf("%3d  ", cooHost[begin + i]);
		}
		printf("\n\n");
	}

	return 0;
}
