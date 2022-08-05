/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2016 Marco Hutter - http://www.jcuda.org
 */

import DeltaOptimizer.BaseOptimizer;
import DeltaOptimizer.BaseOptimizerInterface;
import function.Fuction;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;



/**
 * An example showing how to use the NVRTC (NVIDIA Runtime Compiler) API
 * to compile CUDA kernel code at runtime.
 */
public class BpNetwork_GPU
{
    double nl;  //学习率
    int in_vector;
    int out_vector;
    double[] output; // 输出
    int[] hidden_arf; // 网络结构
    double[][] hidden_w; //权值
    double[][] hidden_d; //偏

    double[] output_w;
    double[] output_d;

    BaseOptimizerInterface deltaOptimizer;

    int blockSizeX = 1024;

    CUfunction FP_function1;
    CUfunction FP_function2;
    CUfunction FP_function3;
    CUfunction FP_function4_Rule;

    MyCUdeviceptr input_device;
    MyCUdeviceptr output_device;
    MyCUdeviceptr output_w_device;
    MyCUdeviceptr output_d_device;

    MyCUdeviceptr[] hidden_out_device;
    MyCUdeviceptr[] hidden_w_device;
    MyCUdeviceptr[] hidden_d_device;

    MyCUdeviceptr output_delta_device;
    MyCUdeviceptr[] hidden_delta_device;


    Fuction[] Act_function; //每层的激活函数（包扩输出层）
    CUfunction[] cuAct_function; //每层的激活函数（包扩输出层）
    CUfunction[] cuAct_function_Batch;
    CUfunction[] cuAct_device_function;//每层的激活函数导函数（包扩输出层）
    CUfunction LossFunction;
    CUfunction BP_hidden_delta_cufunction; //反向传播函数
    CUfunction  BP_W_delta_cufunction;
    CUfunction  BP_d_delta_cufunction;

    public BpNetwork_GPU(int in_vector, int out_vector, double ng, Fuction act_fuction , int[] hidden_arf) {

        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);


        FP_function1 = compilebySourceCore(SourceCodeLib.FP.FPSourceCode[0],"f1");
        FP_function2 = compilebySourceCore(SourceCodeLib.FP.FPSourceCode[1],"f2");
        FP_function3 = compilebySourceCore(SourceCodeLib.FP.FPSourceCode[2],"f3");

        LossFunction = compilebySourceCore(SourceCodeLib.BP.LossDerivateFunctionCode[0],"MseLoss" );
        BP_hidden_delta_cufunction = compilebySourceCore(SourceCodeLib.BP.hidden_delta_1, "hidden_delta_1");
        BP_W_delta_cufunction = compilebySourceCore(SourceCodeLib.BP.W_delta_Code,"W_delta_Code");
        BP_d_delta_cufunction = compilebySourceCore(SourceCodeLib.BP.d_delta_Code,"d_delta_Code");

        Act_function = new Fuction[hidden_arf.length+1];
        cuAct_function = new CUfunction[hidden_arf.length+1];
        cuAct_function_Batch = new CUfunction[hidden_arf.length+1];
        cuAct_device_function = new CUfunction[hidden_arf.length+1];
        for(int i=0; i<Act_function.length; i++){
            Act_function[i] = act_fuction;
            cuAct_function[i] = compilebySourceCore(act_fuction.SourceCode, act_fuction.name);
            cuAct_function_Batch[i] = compilebySourceCore(SourceCodeLib.FP.FPSourceCodeBatch[act_fuction.id], SourceCodeLib.FP.FPSourceCodeBatchName[act_fuction.id] );
            cuAct_device_function[i] = compilebySourceCore(act_fuction.SourceCode_derivate, act_fuction.SourceCode_derivate_Name);
        }



        //为输入创建指针分配内存
        input_device = new MyCUdeviceptr();

        //为输出创建指针分配内存
        output = new double[out_vector];
        output_device = new MyCUdeviceptr();
        //cuMemAlloc(output_device, (long) out_vector * Sizeof.DOUBLE);

        //为隐藏层输出创建指针
        hidden_out_device = new MyCUdeviceptr[hidden_arf.length];
        hidden_delta_device = new MyCUdeviceptr[hidden_arf.length];
        for (int i=0; i<hidden_arf.length; i++){
            hidden_out_device[i] = new MyCUdeviceptr();
            hidden_delta_device[i] = new MyCUdeviceptr();
        }

        output_delta_device = new MyCUdeviceptr();

        //初始网络
        this.nl = ng;
        this.in_vector = in_vector;
        this.out_vector = out_vector;
        this.hidden_arf = hidden_arf;
        hidden_w = new double[hidden_arf.length][];
        hidden_w[0] = init_w(in_vector, hidden_arf[0]);

        for (int k=1; k<hidden_w.length; k++)
            hidden_w[k] = init_w(hidden_arf[k - 1], hidden_arf[k]);

        hidden_d = new double[hidden_arf.length][];

        for (int i=0; i<hidden_d.length; i++)
            hidden_d[i] = new double[hidden_arf[i]];

        output_w = init_w(hidden_arf[hidden_arf.length-1],out_vector);
        output_d = new double[out_vector];

        //把输出权值拷贝到GPU
        output_w_device = new MyCUdeviceptr(output_w);

        output_d_device = new MyCUdeviceptr(output_d);


        //隐藏层
        hidden_w_device = new MyCUdeviceptr[hidden_arf.length];
        hidden_d_device = new MyCUdeviceptr[hidden_arf.length];
        for (int i=0; i<hidden_arf.length; i++){
            hidden_w_device[i] = new MyCUdeviceptr(hidden_w[i]);

            hidden_d_device[i] = new MyCUdeviceptr(hidden_d[i]);
        }

        deltaOptimizer = new BaseOptimizer();

    }

    private void cuDeviceGet(CUdevice device, int i) {
    }

    //编译
    private CUfunction compilebySourceCore(String code, String voidName){
        nvrtcProgram program1 = new nvrtcProgram();
        nvrtcCreateProgram(program1, code, null, 0, null, null);
        nvrtcCompileProgram(program1, 0, null);


        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx1 = new String[1];
        nvrtcGetPTX(program1, ptx1);
        nvrtcDestroyProgram(program1);

        // Create a CUDA module from the PTX code
        CUmodule module1 = new CUmodule();
        cuModuleLoadData(module1, ptx1[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction cUfunction = new CUfunction();
        cuModuleGetFunction(cUfunction, module1, voidName);

        return cUfunction;
    }


    /***
     * @param hostInput = { i1, i2, i3, ... , in }
     * @param host_W =  { w00, w01, w02,...,W0n, w10, w11, w12, .. w1n, ... ,wn0, wn1, ... , Wnm }
     * @return output
     */
    /*
     memPitch=2147483647
     maxThreadsPerBlock=1024
     maxThreadsDim=[1024, 1024, 64]
     maxGridSize=[2147483647, 65535, 65535]
    */


    /**
     * @param in_vector_t 输入维度
     * @param w_len 权值数组长度
     * @param o_len 输出维度
     * @param input_device_t 输入
     * @param w_device_t 权值
     * @param d_device_t 偏置值
     * @param cuAct_function 激活函数
     * @return 输出

    private CUdeviceptr FP_1(int in_vector_t, int w_len, int o_len, CUdeviceptr input_device_t, CUdeviceptr w_device_t , CUdeviceptr d_device_t, CUfunction cuAct_function){
        CUdeviceptr deviceOutput1 = new CUdeviceptr();
        cuMemAlloc(deviceOutput1, (long) w_len * Sizeof.DOUBLE);


        int gridSizeX = (w_len + blockSizeX - 1) / blockSizeX; //max = 1048576*2// 2147483647;

        Pointer p01 = Pointer.to(
                Pointer.to(new int[]{in_vector_t}),
                Pointer.to(new int[]{w_len}),
                Pointer.to(input_device_t),
                Pointer.to(w_device_t),
                Pointer.to(deviceOutput1)
        );
        cuLaunchKernel(FP_function1,
                gridSizeX, 1,1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                p01, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        gridSizeX = (w_len + blockSizeX - 1) / blockSizeX;

        CUdeviceptr output_device_t = new CUdeviceptr();
        cuMemAlloc(output_device_t, o_len * Sizeof.DOUBLE);

        Pointer p02 = Pointer.to(
                Pointer.to(new int[]{in_vector_t}),
                Pointer.to(new int[]{o_len}),
                Pointer.to(deviceOutput1),
                Pointer.to(d_device_t),
                Pointer.to(output_device_t)
        );
        cuLaunchKernel(FP_function2,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                p02, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        cuMemFree(deviceOutput1);


        Pointer kernelParameters3 = Pointer.to(
                Pointer.to(new int[]{o_len}),
                Pointer.to(output_device_t)
        );

        cuLaunchKernel(cuAct_function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters3, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        return output_device_t;
    }

 */



    /**
     * @param lastInput_Batch_Size : batch-Size个输入组成的 input_n*batch-Size 矩阵
     * @param input_n : 一个输入维度
     * @param Batch_Size :
     * @param w : 当层所有权值矩阵 input*N （N为该层节点数）
     * @param d : 偏置值矩阵 N*1
     * @return 节点输出矩阵 N*Batch-Size
      */
    public double[] FP_Batch(double[] lastInput_Batch_Size, int input_n , int Batch_Size, double[] w, double[] d){

        CUdeviceptr W = new CUdeviceptr();
        cuMemAlloc(W, (long) w.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(W,Pointer.to(w), (long) w.length * Sizeof.DOUBLE);

        int out_len = Batch_Size * w.length;

        CUdeviceptr lastInput_Batch_Size_device = new CUdeviceptr();
        cuMemAlloc(lastInput_Batch_Size_device, (long) lastInput_Batch_Size.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(lastInput_Batch_Size_device, Pointer.to(lastInput_Batch_Size), (long) lastInput_Batch_Size.length * Sizeof.DOUBLE);

        CUdeviceptr out_device = new CUdeviceptr();
        cuMemAlloc(out_device, (long) out_len * Sizeof.DOUBLE);

        //int input_len, int W_len ,int out_len, double *X, double *W, double *out
        Pointer pointer1 = Pointer.to(
                Pointer.to(new int[]{ input_n  }),
                Pointer.to(new int[]{ w.length }),
                Pointer.to(new int[]{ out_len  }),
                Pointer.to(lastInput_Batch_Size_device),
                Pointer.to(W),
                Pointer.to(out_device)
        );





        int gridSizeX = ( out_len + blockSizeX - 1) / blockSizeX; //max =  2147483647;

        cuLaunchKernel(FP_function3,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer1, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        cuMemFree(lastInput_Batch_Size_device);
        cuMemFree(W);

        //cuMemcpyDtoH(Pointer.to(out), out_device , out_len * Sizeof.DOUBLE);


        double[] out = new double[Batch_Size * w.length/input_n ];

        CUdeviceptr final_out_device = new CUdeviceptr();
        cuMemAlloc(final_out_device, (long) out.length * Sizeof.DOUBLE);

        CUdeviceptr d_device = new CUdeviceptr();
        cuMemAlloc(d_device, (long) d.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(d_device, Pointer.to(d), (long) d.length * Sizeof.DOUBLE);

        //f4_Relu(int input_len, int d_len ,int out_len, double *input, double *d, double *out)
        Pointer pointer2 = Pointer.to(
                Pointer.to(new int[]{ input_n  }),
                Pointer.to(new int[]{ d.length }),
                Pointer.to(new int[]{ out.length  }),
                Pointer.to(out_device),
                Pointer.to(d_device),
                Pointer.to(final_out_device)
        );

        gridSizeX = ( out.length + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel(FP_function4_Rule,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer2, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(out), final_out_device, (long) out.length * Sizeof.DOUBLE);

        cuMemFree(out_device);
        cuMemFree(d_device);
        cuMemFree(final_out_device);

        return out;
    }






    /**
     * 一层前向传播
     * @param input_n 一数据个输入维度
     * @param Batch_Size :
     * @param w_len 当层所有权值矩阵长度 input_n*N （N为该层节点数）
     * @param d_len 偏置值矩阵长度 N*1
     * @param lastInput_Batch_Size_device 输入矩阵
     * @param W 当层所有权值矩阵 input_n*N （N为该层节点数）
     * @param d 偏置值矩阵 N*1
     * @param fp_actFunction_Batch  激活函数类型
     * @param out 结果矩阵 output_n * Batch_Size
     */
    private void FP_Batch(
            int input_n ,
            int Batch_Size,
            int w_len,
            int d_len,
            MyCUdeviceptr lastInput_Batch_Size_device,
            MyCUdeviceptr W,
            MyCUdeviceptr d,
            CUfunction fp_actFunction_Batch,
            MyCUdeviceptr out
    ){

        int out_len = Batch_Size * w_len;

        CUdeviceptr out_device = new CUdeviceptr();
        cuMemAlloc(out_device, (long) out_len * Sizeof.DOUBLE);

        //int input_len, int W_len ,int out_len, double *X, double *W, double *out
        Pointer pointer1 = Pointer.to(
                Pointer.to(new int[]{ input_n  }),
                Pointer.to(new int[]{ w_len }),
                Pointer.to(new int[]{ out_len  }),
                Pointer.to( lastInput_Batch_Size_device.cUdeviceptr ),
                Pointer.to( W.cUdeviceptr ),
                Pointer.to(out_device)
        );

        int gridSizeX = ( out_len + blockSizeX - 1) / blockSizeX; //max =  2147483647;

        cuLaunchKernel(FP_function3,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer1, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        //cuMemcpyDtoH(Pointer.to(out), out_device , out_len * Sizeof.DOUBLE);


        out_len = Batch_Size * w_len/input_n;

        //CUdeviceptr final_out_device = new CUdeviceptr();
        //cuMemAlloc(final_out_device, (long) out_len * Sizeof.DOUBLE);
        out.malloc((long) out_len * Sizeof.DOUBLE);


        //f4_Relu(int input_len, int d_len ,int out_len, double *input, double *d, double *out)
        Pointer pointer2 = Pointer.to(
                Pointer.to(new int[]{ input_n  }),
                Pointer.to(new int[]{ d_len }),
                Pointer.to(new int[]{ out_len  }),
                Pointer.to(out_device),
                Pointer.to( d.cUdeviceptr ),
                Pointer.to(out.cUdeviceptr)
        );

        gridSizeX = (out_len + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel(fp_actFunction_Batch,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer2, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        cuMemFree(out_device);

    }


    private void FP_GPU_Batch_(int Batch_Size, double[] X_input) {
        int X_size = X_input.length * Sizeof.DOUBLE;


        input_device.malloc(X_size);
        input_device.cpyHtoD(Pointer.to(X_input));


        //cuMemFree(hidden_out_device[0]);
        FP_Batch(in_vector, Batch_Size, hidden_w[0].length, hidden_d[0].length, input_device, hidden_w_device[0], hidden_d_device[0], cuAct_function_Batch[0],hidden_out_device[0]);

       for (int i=1; i<hidden_arf.length; i++) {
          // cuMemFree(hidden_out_device[i]);
           FP_Batch(hidden_arf[i-1], Batch_Size, hidden_w[i].length, hidden_d[i].length, hidden_out_device[i-1], hidden_w_device[i], hidden_d_device[i], cuAct_function_Batch[i],hidden_out_device[i]);
       }
       //cuMemFree(output_device);
        FP_Batch(hidden_arf[hidden_arf.length-1], Batch_Size, output_w.length, output_d.length, hidden_out_device[hidden_arf.length-1], output_w_device, output_d_device, cuAct_function_Batch[hidden_arf.length],output_device);

    }

    public double[] FP_GPU_Batch(int Batch_Size, double[] X_input){
        FP_GPU_Batch_(Batch_Size,X_input);

        int out_size = out_vector*Batch_Size;
        double[] out = new double[out_size];
        //cuMemcpyDtoH(Pointer.to(out), output_device,(long) out_size * Sizeof.DOUBLE);
        output_device.cpyDtoH(Pointer.to(out));

        return out;
    }

























    /**
     *
     * @return 损失函数求导值
     */
    private void BP_GPU_Batch_Loss_d(double[] target){

        long Size = (long) target.length * Sizeof.DOUBLE;
        output_delta_device.malloc(Size);

        CUdeviceptr target_device = new CUdeviceptr();
        cuMemAlloc(target_device, Size);
        cuMemcpyHtoD(target_device,Pointer.to(target), Size);

        //MseLoss(int len, double *t, double *y, double *out)
        Pointer pointer = Pointer.to(
                Pointer.to(new int[]{target.length}),
                Pointer.to(target_device),
                Pointer.to(output_device.cUdeviceptr),
                Pointer.to(output_delta_device.cUdeviceptr) //结果
        );

        int gridSizeX = ( target.length + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel(LossFunction,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

    }



    /**
     * 输出层delta计算完成
     * @param len c
     */
    private void BP_GPU_Batch_Delta_output(int len){

        Pointer pointer = Pointer.to(
                Pointer.to(new int[]{len}),
                Pointer.to(output_device.cUdeviceptr),
                Pointer.to(output_delta_device.cUdeviceptr)
        );

        int gridSizeX = ( len + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel(cuAct_device_function[hidden_arf.length],
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }



    /**
     * 计算隐藏层梯度
     * @param Batch_Size bs
     * @param len 要更新的层长度
     * @param nextNeuer_len 下一层长度
     * @param Delta 要更新层的delta
     * @param nextDelta 下一层delta
     * @param nextW 下一层W
     * @param out 当层输出
     * @param act_d_function 当层激活函数导
     */
    private void BP_GPU_Batch_Delta_hidden(int Batch_Size, int len ,int nextNeuer_len, MyCUdeviceptr Delta, MyCUdeviceptr nextDelta, MyCUdeviceptr nextW, MyCUdeviceptr out, CUfunction act_d_function) {

         int lenXBatch_Size = Batch_Size * len;

        Delta.malloc((long) lenXBatch_Size * Sizeof.DOUBLE);


         //hidden_delta_1(int lenXBatch_Size, int len, int next_len ,double *Delta, double *nextW, double *nextDelta)
         Pointer pointer = Pointer.to(
                 Pointer.to(new int[]{ lenXBatch_Size }),
                 Pointer.to(new int[]{ len }),
                 Pointer.to(new int[]{ nextNeuer_len }),
                 Pointer.to(Delta.cUdeviceptr),
                 Pointer.to(nextW.cUdeviceptr),
                 Pointer.to(nextDelta.cUdeviceptr)
         );


        int gridSizeX = ( lenXBatch_Size + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel(BP_hidden_delta_cufunction,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();


        Pointer pointer1 = Pointer.to(
                Pointer.to(new int[]{ lenXBatch_Size }),
                Pointer.to(out.cUdeviceptr),
                Pointer.to(Delta.cUdeviceptr)
        );

        cuLaunchKernel(act_d_function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer1, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

    }




    /**
     * 计算W的梯度
     * @param Batch_Size bs
     * @param len 要更新的层长度
     * @param last_len 上一层长度
     * @param w_len 要更新的层所有w数量
     * @param Delta 要更新的层的梯度
     * @param last_out 上一层输出
     * @return w的梯度
     */
    private double[] BP_GPU_Batch_Delta_W(int Batch_Size, int len, int last_len, int w_len, MyCUdeviceptr Delta, MyCUdeviceptr last_out){
        double[] delta_w = new double[w_len];
        CUdeviceptr cu_delta_w = new CUdeviceptr();
        cuMemAlloc(cu_delta_w, (long) w_len * Sizeof.DOUBLE);

        //W_delta_Code(int Batch_Size, int len, int last_len ,int w_len ,double *delta, double *last_out, double *delta_w)
        Pointer pointer = Pointer.to(
                Pointer.to(new int[]{ Batch_Size }),
                Pointer.to(new int[]{ len }),
                Pointer.to(new int[]{ last_len }),
                Pointer.to(new int[]{ w_len }),
                Pointer.to( Delta.cUdeviceptr ),
                Pointer.to( last_out.cUdeviceptr ),
                Pointer.to( cu_delta_w )
        );

        int gridSizeX = ( w_len + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel( BP_W_delta_cufunction,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(delta_w),cu_delta_w, (long) w_len * Sizeof.DOUBLE);
        cuMemFree(cu_delta_w);

        return delta_w;
    }



    /**
     * 计算d的梯度
     * @param Batch_Size bs
     * @param len 要更新的层长度
     * @param Delta 要更新的层的梯度
     * @return d的梯度
     */
    private double[] BP_GPU_Batch_Delta_d(int Batch_Size, int len, MyCUdeviceptr Delta){
        double[] delta_d = new double[len];

        CUdeviceptr cu_delta_d = new CUdeviceptr();
        cuMemAlloc(cu_delta_d, (long) len * Sizeof.DOUBLE);

        //d_delta_Code(int Batch_Size,int len, double *delta, double *delta_d)
        Pointer pointer = Pointer.to(
              Pointer.to(new int[]{ Batch_Size }),
              Pointer.to(new int[]{ len }),
              Pointer.to(Delta.cUdeviceptr),
              Pointer.to(cu_delta_d)
        );

        int gridSizeX = ( len + blockSizeX - 1) / blockSizeX;

        cuLaunchKernel(BP_d_delta_cufunction ,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                pointer, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(delta_d),cu_delta_d,(long) len * Sizeof.DOUBLE);
        cuMemFree(cu_delta_d);

        return delta_d;
    }


    private void upgrade(int Batch_Size){
       int i = 0;
       int l = 0;
       //输出层
       double[] output_d_delta = BP_GPU_Batch_Delta_d(
               Batch_Size,
               out_vector,
               output_delta_device
       );
      // System.out.println("out_d_delta   " + Arrays.toString(output_d_delta));
       for (i=0; i < output_d.length; i++)
           output_d[i] += nl * deltaOptimizer.DELTA(output_d_delta[i]);
       //cuMemcpyHtoD(output_d_device, Pointer.to(output_d), (long) output_d.length * Sizeof.DOUBLE);
        output_d_device.cpyHtoD(Pointer.to(output_d));

       // System.out.println( "out_w   " + Arrays.toString(output_w));
        double[] output_w_delta = BP_GPU_Batch_Delta_W(
                Batch_Size,
                out_vector,
                hidden_arf[hidden_arf.length-1],
                output_w.length,
                output_delta_device,
                hidden_out_device[hidden_out_device.length-1]
        );

      // System.out.println( "out_w_delta   " + Arrays.toString(output_w_delta));
        for (i=0; i < output_w.length; i++)
           output_w[i] += nl * deltaOptimizer.DELTA(output_w_delta[i]);
       //cuMemcpyHtoD(output_w_device,Pointer.to(output_w), (long) output_w.length * Sizeof.DOUBLE);
        output_w_device.cpyHtoD(Pointer.to(output_w));


       //最后隐藏层
       double[] hidden_d_end_delta = BP_GPU_Batch_Delta_d(
               Batch_Size,
               hidden_arf[hidden_arf.length-1],
               hidden_delta_device[hidden_delta_device.length-1]
       );
       l = hidden_d_end_delta.length;
       int tl = hidden_d.length-1;
       for (i=0; i < l; i++)
           hidden_d[tl][i] += nl * deltaOptimizer.DELTA(hidden_d_end_delta[i]);
       //cuMemcpyHtoD(hidden_d_device[tl], Pointer.to(hidden_d[tl]), (long) l * Sizeof.DOUBLE);
        hidden_d_device[tl].cpyHtoD(Pointer.to(hidden_d[tl]));


       double[] hidden_w_end_delta = BP_GPU_Batch_Delta_W(
               Batch_Size,
               hidden_arf[hidden_arf.length-1],
               hidden_arf[hidden_arf.length-2],
               hidden_w[hidden_w.length-1].length,
               hidden_delta_device[hidden_delta_device.length -1],
               hidden_out_device[hidden_out_device.length-2]
       );
      // System.out.println("hidden_w "+ (hidden_arf.length-1) + "   " + Arrays.toString(hidden_w_end_delta));
       l = hidden_w_end_delta.length;
       for (i=0; i < l; i++)
            hidden_w[tl][i] += nl * deltaOptimizer.DELTA( hidden_w_end_delta[i] );
      // cuMemcpyHtoD(hidden_w_device[tl], Pointer.to(hidden_w[tl]),(long) l * Sizeof.DOUBLE );
        hidden_w_device[tl].cpyHtoD(Pointer.to(hidden_w[tl]));




       //隐藏层
       for(i = hidden_w.length - 2; i > 0; i--){
           int j = 0;
           double[] d_delta = BP_GPU_Batch_Delta_d(
                   Batch_Size,
                   hidden_arf[i],
                   hidden_delta_device[i]
           );
           for(j = 0; j < d_delta.length; j++)
               hidden_d[i][j] += nl * deltaOptimizer.DELTA( d_delta[j] );
          // cuMemcpyHtoD(hidden_d_device[i], Pointer.to(hidden_d[i]) , (long) d_delta.length * Sizeof.DOUBLE);
           hidden_d_device[i].cpyHtoD( Pointer.to(hidden_d[i]) );

           double[] W_delta = BP_GPU_Batch_Delta_W(
                   Batch_Size,
                   hidden_arf[i],
                   hidden_arf[i-1],
                   hidden_w[i].length,
                   hidden_delta_device[i],
                   hidden_out_device[i-1]
           );


           for(j = 0; j < W_delta.length; j++)
               hidden_w[i][j] += nl * deltaOptimizer.DELTA( W_delta[j] );
           //cuMemcpyHtoD(hidden_w_device[i], Pointer.to(hidden_w[i]), (long) W_delta.length * Sizeof.DOUBLE);
           hidden_w_device[i].cpyHtoD(Pointer.to(hidden_w[i]));
       }


       //输入层
        double[] d_delta = BP_GPU_Batch_Delta_d(
                Batch_Size,
                hidden_arf[0],
                hidden_delta_device[0]
        );
        for(i = 0; i < d_delta.length; i++)
            hidden_d[0][i] += nl * deltaOptimizer.DELTA( d_delta[i] );
        //cuMemcpyHtoD(hidden_d_device[0], Pointer.to(hidden_d[0]) , (long) d_delta.length * Sizeof.DOUBLE);
        hidden_d_device[0].cpyHtoD( Pointer.to(hidden_d[0]) );

        double[] W_delta = BP_GPU_Batch_Delta_W(
                Batch_Size,
                hidden_arf[0],
                in_vector,
                hidden_w[0].length,
                hidden_delta_device[0],
                input_device
        );
       // System.out.println("hidden_w "+ 0  + "   " + Arrays.toString(W_delta));
        for(i = 0; i < W_delta.length; i++)
            hidden_w[0][i] += nl * deltaOptimizer.DELTA( W_delta[i] );
        //cuMemcpyHtoD(hidden_w_device[0], Pointer.to(hidden_w[0]), (long) W_delta.length * Sizeof.DOUBLE);
        hidden_w_device[0].cpyHtoD(Pointer.to(hidden_w[0]));

    }

    public void BP_GPU(double[] input, double[] target, int Batch_Size){

        Arrays.toString(FP_GPU_Batch(Batch_Size, input));

        BP_GPU_Batch_Loss_d(target);

        double[] var = new double[Batch_Size * out_vector];
        //cuMemcpyDtoH(Pointer.to(var) , output_delta_device, (long) var.length * Sizeof.DOUBLE);
       // System.out.println( "out_delta  " + Arrays.toString(var));

        BP_GPU_Batch_Delta_output(Batch_Size * out_vector);


        BP_GPU_Batch_Delta_hidden(
                Batch_Size,
                hidden_arf[hidden_arf.length-1],
                out_vector,
                hidden_delta_device[hidden_arf.length-1],
                output_delta_device,
                output_w_device,
                hidden_out_device[hidden_arf.length-1],
                cuAct_device_function[hidden_arf.length-1]
        );


        for (int i = hidden_arf.length - 2; i >= 0; i--){
            int index = i + 1;
            BP_GPU_Batch_Delta_hidden(
                    Batch_Size,
                    hidden_arf[i],
                    hidden_arf[index],
                    hidden_delta_device[i],
                    hidden_delta_device[index],
                    hidden_w_device[index],
                    hidden_out_device[i],
                    cuAct_device_function[i]
            );

        }

        upgrade(Batch_Size);

      }










    private double[] FP_CPU(double[] input){
       double[][] hidden_out = new double[hidden_w.length][];

       //输入层
        hidden_out[0] = new double[hidden_arf[0]];
        for ( int i = 0; i < hidden_arf[0]; i++){
            for ( int j = 0; j < in_vector; j++)
                 hidden_out[0][i] += input[j] * hidden_w[0][ i*in_vector + j];
            hidden_out[0][i] -= hidden_d[0][i];
            hidden_out[0][i] = Act_function[0].f(hidden_out[0][i]);
        }
        //System.out.println("hidden_out[0] " + Arrays.toString(hidden_out[0]));

        for (int i = 1; i < hidden_w.length; i++) {
            hidden_out[i] = new double[hidden_arf[i]];
            for (int j = 0; j < hidden_arf[i]; j++) {
                for (int k = 0; k < hidden_arf[i - 1]; k++)
                    hidden_out[i][j] += hidden_out[i - 1][k] * hidden_w[i][j * hidden_arf[i - 1] + k];
                hidden_out[i][j] -= hidden_d[i][j];
                hidden_out[i][j] = Act_function[i].f(hidden_out[i][j]);
            }
           // System.out.println("hidden["+i+"] " + Arrays.toString(hidden_out[i]));
        }

        int len = hidden_out.length - 1;
       double[] out = new double[out_vector];
        for ( int i = 0; i <out_vector; i++){
            for ( int j = 0; j < hidden_arf[len]; j++)
                out[i] += hidden_out[len][j] * output_w[ i* hidden_arf[len] + j];
            out[i] -= output_d[i];
            out[i] = Act_function[hidden_arf.length].f(out[i]);
        }

        return out;
    }



    private void BP_CPU(){

    }

    /**
     * 初始化权值
     * @param input_n 输入维度
     * @param neuer_n 该层神经元数量
     * @return  权值矩阵
     */
    private double[] init_w(int input_n,int neuer_n) {
        double[] w = new double[ input_n * neuer_n ];
       for (int i=0;i < neuer_n; i++){
            double[] wi = new double[input_n];
            double sum = 0;
            for (int j = 0; j<wi.length; j++)
               sum += wi[j] = Math.random();

            for (int j = 0; j<wi.length; j++)
                w[i*input_n + j] = 1.0;// wi[j] / sum;
        }
        return  w;
    }

    /**
     * @return 获取可用显存大小 MB
     */
    public int getGpuEnableMemory(){
        int memory = 0;
        String line;
        try {
            Process process =  Runtime.getRuntime().exec("nvidia-smi.exe");
            process.getOutputStream().close();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            while (null != (line = reader.readLine())) {
                if(line.contains("MiB")) break;
            }
            reader.close();

            if (line == null) return memory;

            line = line.substring(1,line.length()-1);
            line = line.substring( line.indexOf("|")+1, line.lastIndexOf("|") );

            char[] chars = line.toCharArray();
            line = "";
            boolean b1 = false;
            for(char c:chars){
                if( c>='0' && c<='9'){
                    line += c;
                    b1 = true;
                }else {
                    if(b1) line += "|" ;
                    b1 = false;
                }
            }

            line = line.substring(1,line.length()-1);

            memory = Integer.parseInt( line.substring(line.indexOf("|")+1) ) - Integer.parseInt( line.substring(0,line.indexOf("|")) );
        } catch (IOException e) {
            e.printStackTrace();
        }
        return memory;
    }
}