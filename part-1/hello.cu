#include <stdio.h>
#include <stdlib.h>
 
// __global__ keyword specifies a device kernel function
__global__ void kernelA(){
    printf("Hello, from the GPU!\n");
}
 
int main()
{
    printf("Hello, from the CPU!\n");
     
    // Set which device should be used
    // The code will default to 0 if not called though
    cudaSetDevice(0);
 
    // Call a device function from the host: a kernel launch
    // Which will print from the device
    kernelA <<<1,1>>>();
 
    // This call waits for all of the submitted GPU work to complete
    cudaDeviceSynchronize();
 
   return 0;
}
