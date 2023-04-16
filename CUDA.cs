using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

class Program
{
    static void Main(string[] args)
    {
        int n = 1024;
        float[] a = new float[n];
        float[] b = new float[n];
        float[] c = new float[n];

        // Initialize arrays
        for (int i = 0; i < n; i++)
        {
            a[i] = i;
            b[i] = i;
        }

        // Initialize CUDA context
        CudaContext context = new CudaContext();

        // Allocate device memory
        CudaDeviceVariable<float> d_a = a;
        CudaDeviceVariable<float> d_b = b;
        CudaDeviceVariable<float> d_c = c;

        // Launch kernel on device
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        dim3 blockSize = new dim3(threadsPerBlock, 1, 1);
        dim3 gridSize = new dim3(blocksPerGrid, 1, 1);
        AddVectorsKernel<<<gridSize, blockSize>>>(d_a.DevicePointer, d_b.DevicePointer, d_c.DevicePointer, n);

        // Copy results from device to host
        d_c.CopyToHost(c);

        // Print results
        for (int i = 0; i < n; i++)
        {
            Console.WriteLine(c[i]);
        }
    }

    static void AddVectorsKernel(CudaDeviceVariable<float> a, CudaDeviceVariable<float> b, CudaDeviceVariable<float> c, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n)
        {
            c[idx] = a[idx] + b[idx];
        }
    }
}
