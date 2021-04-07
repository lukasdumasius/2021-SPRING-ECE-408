// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define BLOCK_SIZE2 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

 
// adds values in Auxillary Array to each output value to finalize them to correct outputs
__global__ void adjustVals(float * inputOutput, float * auxArray, int len){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(( (int)blockIdx.x-1 >= 0) && (i < len)){     // lockIdx.x is an unsigned int, so need to make sure to cast as int
    inputOutput[i] += auxArray[blockIdx.x - 1];
  }
  
}

 //auxArray has size ceil(numElements/(2.0*BLOCK_SIZE))
 // auxOn == true -> write last value to auxillary array (first run)
 // auxOn == false -> dont write last value to auxillary array (second run, when summing up auxillary array)

__global__ void scan(float *input, float *output, int len, bool auxOn) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE]; // contains working sums
  // *** changed above from  2*blockDim.x -> BLOCK_SIZE
  
  
  // loading shared mem
  int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
  if(i < len){
    T[threadIdx.x] = input[i];
  }
  else{
    T[threadIdx.x] = 0.;
  }
  if(i + blockDim.x < len){
    T[threadIdx.x + blockDim.x] = input[i+blockDim.x];
  }
  else{
    T[threadIdx.x + blockDim.x] = 0.;
  }
  
  // Reduction Step
  
  int stride = 1;
  while(stride < 2*blockDim.x){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if(index < 2*blockDim.x && (index-stride) >= 0){
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }
  
  // Post Scan Step (Distribution Tree)
  
  stride = blockDim.x/2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if((index+stride) < 2*blockDim.x){
      T[index+stride] += T[index];
    }
    stride = stride/2;
  }
  
  //copying results back to global mem
  // note that these results are still partial, they need to all be incremented by second kernel
  // after auxillary array is calculated
  __syncthreads();
  if(i < len){
    output[i] = T[threadIdx.x];
  }
  if(i + blockDim.x < len){
    output[i + blockDim.x] = T[threadIdx.x + blockDim.x];
  }
  
  
  // write last value to auxillary array if needed
  __syncthreads();           // This syncthreads is not necessarily needed
  if((auxOn == true) && (threadIdx.x == blockDim.x - 1)){
    input[blockIdx.x] = T[2*blockDim.x - 1];   // we can clobber input and use it as auxillary array for efficiency
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
 
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
 
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(ceil(numElements/(2.0*BLOCK_SIZE)));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  //first scan
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, true);
  cudaDeviceSynchronize();
  
  dim3 dimBlock2(BLOCK_SIZE); //*** changed from BLOCK_SIZE2 -> BLOCK_SIZE
  dim3 dimGrid2( ceil( (ceil(numElements/(2.0*BLOCK_SIZE))) / (2.0*BLOCK_SIZE) ) ); //*** changed second one from BLOCK_SIZE2 -> BLOCK_SIZE
   
  wbLog(TRACE, "I assume that this is 1:  ", ceil( (ceil(numElements/(2.0*BLOCK_SIZE))) / (2.0*BLOCK_SIZE) ) ); // debugging info, we assume that the aux array can be handled in one block
  
  // scan for auxillary array
  scan<<<dimGrid2, dimBlock2>>>(deviceInput, deviceInput, ceil(numElements/(2.0*BLOCK_SIZE)), false);
  cudaDeviceSynchronize();
  
  dim3 dimBlock3(BLOCK_SIZE2);
  dim3 dimGrid3(ceil(numElements/(1.0*BLOCK_SIZE2)));
  
  if(1.0*numElements > 2.0*BLOCK_SIZE){
    // adjust values by adding the auxillary array values to the output
    adjustVals<<<dimGrid3, dimBlock3>>>(deviceOutput, deviceInput, numElements);
    cudaDeviceSynchronize();
  }
  
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
