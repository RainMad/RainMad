#include "./kernel.h"

#include <cstddef>

// per reference ! --> immer per value
// es wird auf einen anderen speicher gegriffen -> es kracht
// es werden immer ein vielfaches von 32 an Thread gestartet -> bei 33 Zeichen werden 64 Threads gestartet
// deswegen wir auch size mit�bergeben

// thread nummer ist relative zum block
// block nummer ist relativ zur grafikkarte
// daraus muss ide absolute threadnummer berechnet werden

// divergenten code vermeiden! -> eine der gr��ten Bremsen
__global__ void kernel(char * const p_dst, char const * const p_src, std::size_t const size) {
	// blockDim Anzahl der Threads pro block
	auto const t{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber

	if (t < size) {
		p_dst[t] = p_src[t];
	}
}

cudaError_t call_kernel(dim3 const big, dim3 const tib, char * const p_dst, char const * const p_src, std::size_t const size) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads
	kernel <<<big, tib>>> (p_dst, p_src, size);
	return cudaGetLastError();
}