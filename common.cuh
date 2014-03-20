// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

#pragma once

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

enum SearchBounds {
	SearchBoundsLower,
	SearchBoundsUpper
};

#define DEVICE __forceinline__ __device__

template<typename T>
cudaError_t copyHtoD(T* dest_global, const T* source_host, size_t count) {
	return cudaMemcpy(dest_global, source_host, sizeof(T) * count,
		cudaMemcpyHostToDevice);
}
template<typename T>
cudaError_t copyHtoD(T* dest_global, const std::vector<T>& source_host) {
	size_t size = source_host.size();
	return size ? copyHtoD(dest_global, &source_host[0], size) : cudaSuccess;
}
template<typename T>
cudaError_t copyDtoH(T* dest_host, const T* source_global, size_t count) {
	return cudaMemcpy(dest_host, source_global, sizeof(T) * count,
		cudaMemcpyDeviceToHost);
}
template<typename T>
cudaError_t copyDtoD(T* dest_global, const T* source_global, size_t count) {
	return cudaMemcpy(dest_global, source_global, sizeof(T) * count,
		cudaMemcpyDeviceToDevice);
}

template<typename T>
cudaError_t cudaMalloc2(T** data, int count) {
	return cudaMalloc((void**)data, sizeof(T) * count);
}
template<typename T>
cudaError_t cudaMalloc2(T** data, const std::vector<T>& source) {
	cudaError_t error = cudaMalloc2(data, (int)source.size());
	if(cudaSuccess == error) {
		error = copyHtoD(*data, source);
		if(cudaSuccess != error)
			cudaFree(*data);
	}
	return error;
}

////////////////////////////////////////////////////////////////////////////////

template<int NT, typename T>
struct CTAReduce {
	enum { Size = NT, Capacity = NT };
	struct Storage { T shared[Capacity]; };

	DEVICE static T Reduce(int tid, T x, Storage& storage) {
		storage.shared[tid] = x;
		__syncthreads();

		// Fold the data in half with each pass.
		#pragma unroll
		for(int destCount = NT / 2; destCount >= 1; destCount /= 2) {
			if(tid < destCount) {
				// Read from the right half and store to the left half.
				x += storage.shared[destCount + tid];
				storage.shared[tid] = x;
			}
			__syncthreads();
		}
		T total = storage.shared[0];
		__syncthreads();
		return total;
	}
};

////////////////////////////////////////////////////////////////////////////////

template<int NT, typename T>
struct CTAScan {
	enum { Size = NT, Capacity = 2 * NT + 1 };
	struct Storage { T shared[Capacity]; };

	DEVICE static T Scan(int tid, T x, Storage& storage, T* total) {

		storage.shared[tid] = x;
		int first = 0;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tid >= offset)
				x += storage.shared[first + tid - offset];
			first = NT - first;
			storage.shared[first + tid] = x;
			__syncthreads();
		}
		*total = storage.shared[first + NT - 1];

		x = tid ? storage.shared[first + tid - 1] : 0;

		__syncthreads();
		return x;
	}

	DEVICE static T Scan(int tid, T x, Storage& storage) {
		T total;
		return Scan(tid, x, storage, &total);
	}
};

////////////////////////////////////////////////////////////////////////////////

template<typename T, typename It>
DEVICE int BinarySearch(It data, int count, T key, SearchBounds bounds) {
	int begin = 0;
	int end = count;
	while(begin < end) {
		int mid = (begin + end) / 2;
		T key2 = data[mid];
		bool pred = (SearchBoundsUpper == bounds) ? !(key < key2) : (key2 < key);
		if(pred) begin = mid + 1;
		else end = mid;
	}
	return begin;
}

template<typename It1, typename It2>
DEVICE int MergePath(It1 a, int aCount, It2 b, int bCount, int diag,
	SearchBounds bounds) {

	typedef typename std::iterator_traits<It1>::value_type T;
	int begin = max(0, diag - bCount);
	int end = min(diag, aCount);

	while(begin < end) {
		int mid = (begin + end)>> 1;
		T aKey = a[mid];
		T bKey = b[diag - 1 - mid];
		bool pred = (SearchBoundsUpper == bounds) ? 
			(aKey < bKey) : !(bKey < aKey);
		if(pred) begin = mid + 1;
		else end = mid;
	}
	return begin;
}
