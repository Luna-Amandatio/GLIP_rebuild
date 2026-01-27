## 一.头文件替换
将**所有**.cu文件中**THC头文件**替换为：
```
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/ThrustAllocator.h>
```
## 对各个cuda文件

### 1. deform_conv_cuda.cu   
    无需更改

### 2. deform_conv_kernel_cuda.c
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

-------------------------------------------------------------------------

第351行修改前：
atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);

第351行修改后：
atomic_add(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);

-------------------------------------------------------------------------

第709行修改前：
atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);

第709行修改后：
atomic_add(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
```

### 3. deform_pool_cuda.cu
    无需更改
### 4. deform_pool_kernel_cuda.cu
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

------------------------------------------------------------------------------------------

第258-261行修改前：
atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);

第258-261行修改后：
atomic_add(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
atomic_add(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
atomic_add(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
atomic_add(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);

------------------------------------------------------------------------------------------

第276-277行修改前：
atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);
atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);

第276-277行修改后：
atomic_add(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);
atomic_add(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);       
```
### 5. ml_nms.cu
```cpp
第71行修改前：
const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);


const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;

----------------------------------------------------------------------------------------------

第86行修改前：
const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

第86行修改后：
const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;

----------------------------------------------------------------------------------------------

第90行修改前：
THCState *state = at::globalContext().lazyInitCUDA();

第90行修改后：
删去

----------------------------------------------------------------------------------------------

第96行修改前：
mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

第96行修改后：
mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));

----------------------------------------------------------------------------------------------

第98-99行修改前：
dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
              
第98-99行修改后：
dim3 blocks(col_blocks, col_blocks);

----------------------------------------------------------------------------------------------

第107行修改前：
THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

第107行修改后：
C10_CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

----------------------------------------------------------------------------------------------

第132行修改前：
THCudaFree(state, mask_dev);

第132行修改后：
c10::cuda::CUDACachingAllocator::raw_delete((void*)mask_dev);
```
### 6. nms.cu
```cpp
第66行修改前：
const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);

第66行修改后：
const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;

---------------------------------------------------------------------------

第81行修改前：
const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

第81行修改后：
const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;

---------------------------------------------------------------------------

第85行修改前：
THCState *state = at::globalContext().lazyInitCUDA();

第85行修改后：
删去

---------------------------------------------------------------------------

第90-91行修改前：
dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
            THCCeilDiv(boxes_num, threadsPerBlock));

第90-91行修改后：  
dim3 blocks(col_blocks, col_blocks);

--------------------------------------------------------------------------

第99行修改前：
THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

第99行修改后：
cudaMemcpy(&mask_host[0],
             mask_dev,
             sizeof(unsigned long long) * boxes_num * col_blocks,
             cudaMemcpyDeviceToHost);
             
--------------------------------------------------------------------------

第124行修改前：
THCudaFree(state, mask_dev);

第124行修改后：
c10::cuda::CUDACachingAllocator::raw_delete((void*)mask_dev);
```

### 7. ROIAlign_cuda.cu
```cpp
第276行修改前：
dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));

第276行修改后：
dim3 grid(std::min(static_cast<long long>(output_size + 511L) / 512L, 4096LL));

--------------------------------------------------------------------------------

第321行修改前：
dim3 grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));

第321行修改后：
dim3 grid(std::min(static_cast<long long>(grad.numel() + 511L) / 512L, 4096LL));

--------------------------------------------------------------------------------

第280,298,326,345行修改前：
THCudaCheck(cudaGetLastError());

第280,298,326,345行修改后：
C10_CUDA_CHECK(cudaGetLastError());
```
### 8. ROIPool_cuda.cu
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
### 9. SigmoidFocalLoss_cuda.cu
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

## 对.py文件

* 使用IDE搜索功能，将**np.float**改为**np.float32**