#ifndef POOLMAN_HPP
#define POOLMAN_HPP

#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <pthread.h>

class PoolMan
{
public:
  PoolMan();
  PoolMan( uint32_t align_pow );
  ~PoolMan();

  void add_pool( void *pool, uint64_t size );

  void *malloc( uint64_t size );
  void  free  ( void    *ptr  );

  void finish_frees();              // Forces completion of any deferred frees
                                    // (really only useful for getting accurate stats)
  uint64_t get_free_bytes();        // Total number of allocable bytes available
  uint64_t get_total_bytes();       // Maximum byte capacity of the pool
  uint32_t get_free_count();        // Number of segments in the free maps
  uint64_t get_max_malloc_bytes();  // Largest malloc that can currently be accommodated
  uint32_t get_alloc_count();       // Current number of allocations in this pool
  uint64_t get_peak_bytes();        // Peak utilization in bytes since creation

  void print_pool();                // For debugging; prints the allocation,
                                    // free_size, and free_addr maps
private:
  void init_();                     // Performs initialization activities common to all constructors
  void free_( void *ptr );          // Updates the free maps to actually free the given pointer
  void finish_frees_();

  uint32_t align_pow_;              // 2^(align_pow_) is the minimum alignment for mallocs
  uint64_t max_possible_alloc;      // Largest allocation this pool can provide

  uint64_t free_bytes_;             // Number of bytes available in the pool
  uint64_t total_bytes_;            // Maximum capacity of the pool
  uint64_t peak_bytes_;             // Peak number of bytes allocated at one time since creation

  // Maps a segment size to an address
  // Using an ordered map for efficient best-fit searching
  // Using a multimap because multiple segments can have the same size
  std::multimap<uint64_t,uintptr_t> free_size_;

  // Maps an address to a segment size
  // Using an ordered map for efficient address overlap searching
  // Using a normal map because addresses are unique
  std::map<uintptr_t,uint64_t> free_addr_;

  // Maps an address to an allocation size
  // Using an unordered map for efficient finds of exact addresses
  std::unordered_map<void*,uint64_t> allocs_;

  pthread_mutex_t malloc_lock;
  pthread_mutex_t frees_lock;

  std::vector<void*> frees;         // Using a vector (vs queue or stack) for debugging
  pthread_cond_t     frees_exist;
};

#endif