import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;

public class MyCUdeviceptr {
    public CUdeviceptr cUdeviceptr;
    public long Size;

    public MyCUdeviceptr(){
        cUdeviceptr = new CUdeviceptr();
        Size = 0;
    }

    public MyCUdeviceptr(long size){
        cUdeviceptr = new CUdeviceptr();
        malloc(size);
    }

    public MyCUdeviceptr(double[] doubles){
        cUdeviceptr = new CUdeviceptr();
        malloc((long) doubles.length * Sizeof.DOUBLE);
        cpyHtoD(Pointer.to(doubles));
    }

    public void malloc(long size){

        if(size != Size)
            if(Size==0){
                Size = size;
                cuMemAlloc(cUdeviceptr, size);
            } else {
                cuMemFree(cUdeviceptr);
                Size = size;
                cuMemAlloc(cUdeviceptr, size);
            }
    }

    public void free(){
        if(Size!=0) {
            Size = 0;
            cuMemFree(cUdeviceptr);
        }
    }




    public void cpyDtoH(Pointer pHost, long size){
        cuMemcpyDtoH(pHost,cUdeviceptr,size);
    }
    public void cpyDtoH(Pointer pHost){
        cpyDtoH(pHost,Size);
    }

    public  void  cpyHtoD(Pointer p, long size){
        cuMemcpyHtoD(cUdeviceptr, p, size);
    }
    public  void  cpyHtoD(Pointer p){
        cuMemcpyHtoD(cUdeviceptr, p, Size);
    }
}
