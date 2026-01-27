## 一.头文件替换
将所有.cu文件中THC头文件替换为：
```
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/ThrustAllocator.h>
```

## 二.内存分配修改
```
// 旧代码
mask_dev = (unsigned long long*) THCudaMalloc(state, size);

// 新代码（CUDA12.8通用）
mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(size);
```

# 三.错误检查更新
```
// 旧代码
THCudaCheck(cudaGetLastError());
// 新代码
C10_CUDA_CHECK(cudaGetLastError());
```
## 对各个cuda文件

1. deform_conv_cuda.cu

2. deform_conv_kernel_cuda.c
```cpp
在文件头部添加

template<typename T>
__device__ void atomic_add(T* address, T val) {
    if constexpr (std::is_same_v<T, at::Half>) {
        // c10::Half 需要转换为 float 原子加法
        atomicAdd(reinterpret_cast<float*>(address), __half2float(val));
    } else {
        // float/double 直接原生支持
        atomicAdd(address, val);
    }
}
------------------------------------------------------
351
atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
351
atomic_add(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);

--------------------------------------------------------
709
atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
709
atomic_add(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
----------------------------------------------------------
```
3. deform_pool_cuda.cu

4. deform_pool_kernel_cuda.cu
```cpp
在文件头部添加

template<typename T>
__device__ void atomic_add(T* address, T val) {
    if constexpr (std::is_same_v<T, at::Half>) {
        // c10::Half 需要转换为 float 原子加法
        atomicAdd(reinterpret_cast<float*>(address), __half2float(val));
    } else {
        // float/double 直接原生支持
        atomicAdd(address, val);
    }
}
------------------------------------------------------
258-261
atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);

atomic_add(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
atomic_add(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
atomic_add(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
atomic_add(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);
------------------------------------------------------------------------------------------------
276-277
atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);
atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);
        
atomic_add(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);
atomic_add(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);       
```
5. ml_nms.cu
```cpp
90,
THCState *state = at::globalContext().lazyInitCUDA();
删去
------------------------------------------------------
107
THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

C10_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

--------------------------------------------------------
132
THCudaFree(state, mask_dev);

c10::cuda::CUDACachingAllocator::raw_delete((void*)mask_dev);
---------------------------------------------------------
96
mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));
-------------------------------------------------------------------------
71，86
const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;
---------------------------------------------------------------------
98，99
dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));

dim3 blocks(col_blocks, col_blocks);
```
6. nms.cu
```cpp
85
THCState *state = at::globalContext().lazyInitCUDA();
删去
------------------------------------------------------

99
THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

cudaMemcpy(&mask_host[0],
             mask_dev,
             sizeof(unsigned long long) * boxes_num * col_blocks,
             cudaMemcpyDeviceToHost);
--------------------------------------------------------
124
THCudaFree(state, mask_dev);

c10::cuda::CUDACachingAllocator::raw_delete((void*)mask_dev);
-------------------------------------------------------------
66
const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
----------------------------------------------------------
81
const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);
const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;
-----------------------------------------------------------
90-91
dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
            THCCeilDiv(boxes_num, threadsPerBlock));
            
dim3 blocks(col_blocks, col_blocks);
```
7. ROIAlign_cuda.cu
```cpp
第276行修改前：
dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));

第276行修改后：
dim3 grid(std::min(static_cast<long long>(output_size + 511L) / 512L, 4096LL));

--------------------------------------------------------------
第321行修改前：
dim3 grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));

第321行修改后：
dim3 grid(std::min(static_cast<long long>(grad.numel() + 511L) / 512L, 4096LL));

--------------------------------------------------------------
第280,298,326,345行修改前：
THCudaCheck(cudaGetLastError());

第280,298,326,345行修改后：
C10_CUDA_CHECK(cudaGetLastError());
```
8. ROIPool_cuda.cu
```cpp
第130行修改前：
dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));

第130行修改后：
dim3 grid(std::min((output_size + 511LL) / 512LL, 4096LL));

---------------------------------------------------------------------------------
第177行修改前：
dim3 grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));

第177行修改后： 
dim3 grid(std::min((grad.numel() + 511LL) / 512LL, 4096LL));
---------------------------------------------------------------------------------
第134,152,182,201行修改前：
THCudaCheck(cudaGetLastError());

第134,152,182,201行修改后：
C10_CUDA_CHECK(cudaGetLastError());
```
9. SigmoidFocalLoss_cuda.cu
```cpp
第121行修改前：
dim3 grid(std::min(THCCeilDiv(losses_size, 512L), 4096L));

第121行修改后：
dim3 grid(std::min((output_size + 511LL) / 512LL, 4096LL));

---------------------------------------------------------------------------------
第165行修改前：
dim3 grid(std::min(THCCeilDiv(d_logits_size, 512L), 4096L));

第165行修改后： 
dim3 grid(std::min((d_logits_size + 511LL) / 512LL, 4096LL));
---------------------------------------------------------------------------------
第125,140,169,186行修改前：
THCudaCheck(cudaGetLastError());

第125,140,169,186行修改后：
C10_CUDA_CHECK(cudaGetLastError());
```

#