// Sean Baxter's GTC talk companion source.
// http://nvlabs.github.io/moderngpu/

// Binary search-driven load-balancing search. Here we use for CSR->COO mapping.
// We first execute a binary search on the CSR array. We then run a sequentially
// merge on exactly VT items from the array of natural numbers. This does not
// allow for empty rows.

#include "common.cuh"

