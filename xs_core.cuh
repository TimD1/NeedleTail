#ifndef XS_CORE_CUH
#define XS_CORE_CUH

#include "nw_general.h"

__global__ void xs_core_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * xf_mat_row0,
  int * xf_mat_row1,
  int * mat
) {
  // Get the global thread index.
  uint32_t g_tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Write 0 to the first cell of our transformed matrix row0.
  if (g_tx == 0)
    xf_mat_row0[0] = 0;
  // Write g_tx * mis_or_ind to the first and
  // second cell of the tranformed matrix row1.
  if (g_tx == 1) {
    xf_mat_row1[0] = g_tx * mis_or_ind;
    xf_mat_row1[1] = g_tx * mis_or_ind;
  }
  // Initialize the border elements of our untransformed matrix.
  if (g_tx <= qlen)
    mat[(tlen + 1) * g_tx] = g_tx * mis_or_ind;
  if (g_tx <= tlen)
    mat[g_tx] = g_tx * mis_or_ind;
}

__global__ void xs_core_comp(
  // Kernel management variables.
  bool wr_q_border_elt, // Do we need to write a border element for query?
  bool wr_t_border_elt, // Do we need to write a border element for target?
  uint32_t comp_w,      // Number of matrix elements we are computing.
  uint32_t comp_x_off,  // What is our x-dimension offset for our compute region?
  uint32_t comp_y_off,  // What is our y-dimension offset for our compute region?
  uint32_t wave_itr,    // Current wavefront iteration.
  // Variables regarding matrix computations.
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * xf_mat_row0,
  int * xf_mat_row1,
  int * xf_mat_row2,
  int * mat
) {
  // Get the global and local thread index.
  uint32_t g_tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint32_t l_tx = threadIdx.x;
  extern __shared__ int smem[];
  int * s_row_up = smem;
  // If we need to write a border element for our query.
  if (g_tx == 0 && wr_q_border_elt)
    xf_mat_row2[0] = (wave_itr + 2) * mis_or_ind;
  // If we need to write a border element for our target.
  if (g_tx == 0 && wr_t_border_elt)
    xf_mat_row2[comp_x_off + comp_w] = (wave_itr + 2) * mis_or_ind;
  // If we are in the compute region.
  if (g_tx >= comp_x_off && g_tx < comp_x_off + comp_w) {
    // Fetch into shared memory.
    if (l_tx == 0 || g_tx == comp_x_off)
      s_row_up[l_tx] = xf_mat_row1[g_tx - 1];
    s_row_up[l_tx + 1] = xf_mat_row1[g_tx];
    __syncthreads();
    // Do the NW cell calculation.
    int match = xf_mat_row0[g_tx - 1]
      + cuda_nw_get_sim(q[comp_y_off - g_tx - 1], t[g_tx - 1]);
    int del = s_row_up[l_tx + 1] + mis_or_ind;
    int ins = s_row_up[l_tx] + mis_or_ind;
    int cell = match > del ? match : del;
    cell = cell > ins ? cell : ins;
    // Write back to our current sliding window row index.
    xf_mat_row2[g_tx] = cell;
    // Write back to our untransformed matrix.
    mat[(tlen + 1) * (comp_y_off - g_tx) + g_tx] = cell;
  }
}

int * xs_t_geq_q_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  void * mem,
  cudaStream_t * stream
) {

  // Maintain a sliding window of 3 rows of our transformed matrix.
  // This is useful because with the transformation matrix we get
  // complete memory coalescing on both reads and writes.
  int * xf_mat_row_temp = NULL;
  int * xf_mat_row0_d = (int *) mem;
  int * xf_mat_row1_d = xf_mat_row0_d + (tlen + 1);
  int * xf_mat_row2_d = xf_mat_row1_d + (tlen + 1);

  // Maintain a full untransformed matrix for PCIe transfer after
  // compute is done. This min/maxes our memory utilization.
  int * mat_d = xf_mat_row2_d + (tlen + 1);

  // Pointers to target and query.
  char * t_d = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  char * q_d = t_d + tlen;

  // Copy our target and query to the GPU.
  cudaMemcpyAsync(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, *stream);

  // Prepare the first 2 rows of our transformed compute matrix,
  // and the border elements for our untranformed matrix.
  uint32_t init_num_threads = (tlen + 1) > (qlen + 1) ? (tlen + 1) : (qlen + 1);
  dim3 init_g_dim(ceil(init_num_threads / ((float) 1024)));
  dim3 init_b_dim(1024);
  xs_core_init <<<init_g_dim, init_b_dim, 0, *stream>>>
    (tlen, qlen, mis_or_ind, xf_mat_row0_d, xf_mat_row1_d, mat_d);

  // Run our matrix scoring algorithm.
  uint32_t comp_num_threads = (tlen + 1);
  dim3 comp_g_dim(ceil(comp_num_threads / ((float) 1024)));
  dim3 comp_b_dim(1024);
  // Kernel management variables, see kernel definiton
  // for comments on what these are used for.
  bool wr_q_border_elt = true;
  bool wr_t_border_elt = true;
  uint32_t comp_w = 0;
  uint32_t comp_x_off = 1;
  uint32_t comp_y_off = 1;
  // Other variables to help manage the kernel manager variables...
  bool square_matrix = (qlen == tlen);
  uint32_t max_comp_w_cnt = 0;
  int8_t comp_w_increment = 1;
  uint32_t max_comp_w = qlen < tlen ? qlen : tlen;
  uint32_t largest_dim = tlen > qlen ? tlen : qlen;
  uint32_t smallest_dim = tlen < qlen ? tlen : qlen;
  uint32_t max_comp_w_cnt_max = largest_dim - smallest_dim + 1;
  // Loop through every wavefront/diagonal.
  for (uint32_t wave_itr = 0; wave_itr < qlen + tlen - 1; ++wave_itr) {
    // Update kernel management variables.
    wr_q_border_elt = (wave_itr + 2 < qlen + 1);
    wr_t_border_elt = (wave_itr + 2 < tlen + 1);
    comp_w += comp_w_increment;
    comp_x_off += (wave_itr + 2 > qlen + 1);
    ++comp_y_off;

    // Launch our kernel.
    xs_core_comp <<<comp_g_dim, comp_b_dim, 1025 * sizeof(int), *stream>>>
      (wr_q_border_elt, wr_t_border_elt, comp_w,
        comp_x_off, comp_y_off, wave_itr, t_d, q_d,
          tlen, qlen, mis_or_ind, xf_mat_row0_d,
            xf_mat_row1_d, xf_mat_row2_d, mat_d);

    // Update other management variables.
    max_comp_w_cnt = comp_w == max_comp_w ?
      max_comp_w_cnt += 1 : max_comp_w_cnt;
    if (square_matrix)
      comp_w_increment = max_comp_w_cnt == 1 ? -1 : 1;
    else
      comp_w_increment = max_comp_w_cnt == 0 ? 1 :
        max_comp_w_cnt == max_comp_w_cnt_max ? -1 : 0;

    // Slide our window in our compute matrix.
    xf_mat_row_temp = xf_mat_row0_d;
    xf_mat_row0_d = xf_mat_row1_d;
    xf_mat_row1_d = xf_mat_row2_d;
    xf_mat_row2_d = xf_mat_row_temp;
  }

  // Copy back our untransformed matrix to the host.
  int * mat = new int [(tlen + 1) * (qlen + 1)];
  cudaMemcpyAsync(mat, mat_d, (tlen + 1) * (qlen + 1) * sizeof(int), cudaMemcpyDeviceToHost, *stream);
  return mat;
}

#endif