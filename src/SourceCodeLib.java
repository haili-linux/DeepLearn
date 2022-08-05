public class SourceCodeLib {

    public final static class FP {
    public final static String[] FPSourceCode = {
            //0
            "extern \"C\"" + "\n" +
                    "__global__ void f1(int input_len, int w_len, double *input, double *w, double *sum)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    //"    int i = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;"+"\n" +
                    "    if (i<w_len)" + "\n" +
                    "    {" + "\n" +
                    "        sum[i] = input[i%input_len] * w[i];" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n"
            ,
            //1
            "extern \"C\"" + "\n" +
                    "__global__ void f2(int input_len, int out_len, double *sum, double *d,double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    //"  int i = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;"+"\n" +
                    "    if (i<out_len)" + "\n" +
                    "    {" + "\n" +
                    "        int index = i * input_len;" + "\n" +
                    "        out[i] = sum[index] + d[i]; //+ sum[index+1] + sum[index+2];" + "\n" +
                    "        int j = index + 1; " + "\n" +
                    "        int m = index + input_len;" + "\n" +
                    "        #pragma unroll                       " + "\n" +
                    "        for(j; j<m; j++) out[i] += sum[j];" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n"
            ,
            //2
            "extern \"C\"" + "\n" +
                    /**
                     * input_len : 每个输入的维度
                     * W_len :  当层权值个数，W的长度，等于 神经元个数*input_len
                     * out_len: Batch_Size * W_len
                     * X : 全部输入 有Batch_Size组
                     * W : 权值
                     * out : 神经元个数 * input_len * Batch_Size
                     */
                    "__global__ void f3(int input_len, int W_len ,int out_len, double *X, double *W, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if ( i <  out_len )" + "\n" +
                    "    {" + "\n" +
                    "       int index0 = i / W_len;  //列           " + "\n" +
                    "       int index1 = i % W_len;  //个           " + "\n" +
                    "       int index2 = index1 % input_len;        " + "\n" +
                    "       int index3 = index0*input_len + index2; " + "\n" +
                    "       out[i] = X[ index3 ] * W[ index1 ];     " + "\n" +
                    "    }" + "\n" +
                    "}" + "\n"
            ,
    };


    public final static String[] FPSourceCodeBatchName = {" ", "fp_Sigmoid", "fp_Tanh", "fp_Relu", "fp_LRelu"};
    public final static String[] FPSourceCodeBatch = {
            //0
            "  "
            ,
            //1 Sigmoid
            "extern \"C\"" + "\n" +
                    "__global__ void fp_Sigmoid(int input_len, int d_len ,int out_len, double *input, double *d, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;    " + "\n" +
                    "    if (i < out_len)                                 " + "\n" +
                    "    {                                                 " + "\n" +
                    "        out[i] = -d[ i%d_len ];                        " + "\n" +
                    "        int j = i * input_len;                        " + "\n" +
                    "        int m = j + input_len;                        " + "\n" +
                    "        #pragma unroll 4                      " + "\n" +
                    "        for( j; j<m; j++) out[i] += input[j];         " + "\n" +
                    "        out[i] = 1.0 / ( 1.0 + exp(-out[i]) );        " + "\n" +
                    "    }                                                 " + "\n" +
                    "}" + "\n"
            ,
            //2 Tanh
            "extern \"C\"" + "\n" +
                    "__global__ void fp_Tanh(int input_len, int d_len ,int out_len, double *input, double *d, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;    " + "\n" +
                    "    if (i < out_len)                                 " + "\n" +
                    "    {                                                 " + "\n" +
                    "        out[i] = -d[ i%d_len ];                        " + "\n" +
                    "        int j = i * input_len;                        " + "\n" +
                    "        int m = j + input_len;                        " + "\n" +
                    "        #pragma unroll 4                             " + "\n" +
                    "        for( j; j<m; j++) out[i] += input[j];         " + "\n" +
                    "        double x1 = exp(out[i]);                      " + "\n" +
                    "        double x2 = exp(-out[i]);                     " + "\n" +
                    "        out[i] = (x1 - x2) / (x1 + x2);               " + "\n" +
                    "    }                                                 " + "\n" +
                    "}" + "\n"
            ,
            //3 Rule
            "extern \"C\"" + "\n" +
                    "__global__ void fp_Relu(int input_len, int d_len ,int out_len, double *input, double *d, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;    " + "\n" +
                    "    if (i < out_len)                                 " + "\n" +
                    "    {                                                 " + "\n" +
                    "        out[i] = -d[ i%d_len ];                        " + "\n" +
                    "        int j = i * input_len;                        " + "\n" +
                    "        int m = j + input_len;            " + "\n" +
                    "        #pragma unroll 4                      " + "\n" +
                    "        for( j; j<m; j++) out[i] += input[j];         " + "\n" +
                    "        if(out[i] < 0) out[i]=0;                      " + "\n" +
                    "    }                                                 " + "\n" +
                    "}" + "\n"
            ,
            // LRelu
            "extern \"C\"" + "\n" +
                    "__global__ void fp_LRelu(int input_len, int d_len ,int out_len, double *input, double *d, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;    " + "\n" +
                    "    if (i < out_len)                                 " + "\n" +
                    "    {                                                 " + "\n" +
                    "        out[i] = -d[ i%d_len ];                        " + "\n" +
                    "        int j = i * input_len;                        " + "\n" +
                    "        int m = j + input_len;                        " + "\n" +
                    "        #pragma unroll 4                               " + "\n" +
                    "        for( j; j<m; j++) out[i] += input[j];         " + "\n" +
                    "        if(out[i] < 0) out[i] = 0.001 * out[i];       " + "\n" +
                    "    }                                                 " + "\n" +
                    "}" + "\n"
    };
}

    public final static class BP{
        public final static String[] LossDerivateFunctionCode = {
                //0 MSELoss
                "extern \"C\"" + "\n" +
                        "__global__ void MseLoss(int len, double *t, double *y, double *out)" + "\n" +
                        "{" + "\n" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " + "\n" +
                        "    if ( i < len )                                 " + "\n" +
                        "    {                                              " + "\n" +
                        "        out[i] = t[i] - y[i];                      " + "\n" +
                        "    }" + "\n" +
                        "}" + "\n"
        };

        public final static String hidden_delta_1 =
               //1 隐藏层梯度计算 1
                "extern \"C\"" + "\n" +
                        "__global__ void hidden_delta_1(int lenXBatch_Size, int len, int next_len ,double *Delta, double *nextW, double *nextDelta)" + "\n" +
                        "{" + "\n" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " + "\n" +
                        "    if ( i < lenXBatch_Size )                                 " + "\n" +
                        "    {                                              " + "\n" +
                        "         Delta[i] = 0;                    " + "\n" +
                        "         int bs = i / len;                    " + "\n" +
                        "         int k = i % len;                   " + "\n" +
                        "         int j = bs * next_len;                    " + "\n" +
                        "         int m = j + next_len;                     " + "\n" +
                        "         #pragma unroll 4                      " + "\n" +
                        "         for( j; j < m; j++ ){              " + "\n" +
                        "           int l = j % next_len;                   " + "\n" +
                        "           Delta[i] += nextDelta[j] * nextW[ l*len + k  ];   " + "\n" +
                        "         }                                  " + "\n" +
                        "                                    " + "\n" +
                        "    }" + "\n" +
                        "}" + "\n";



        public final static String W_delta_Code =
                //len:本层节点数，  last_len:上一层节点数， w_len:本层所有w数量
                "extern \"C\"" + "\n" +
                        "__global__ void W_delta_Code(int Batch_Size, int len, int last_len ,int w_len ,double *delta, double *last_out, double *delta_w)" + "\n" +
                        "{" + "\n" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " + "\n" +
                        "    if ( i < w_len )                                 " + "\n" +
                        "    {                                              " + "\n" +
                        "          delta_w[i] = 0;                          " + "\n" +
                        "          int bs;                                     " + "\n" +
                        "          #pragma unroll 4                    " + "\n" +
                        "          for( bs = 0; bs < Batch_Size; bs++ ){                       " + "\n" +
                        "                delta_w[i] += delta[ bs*len + i/last_len ] * last_out[ bs * last_len + i%last_len ];                   " + "\n" +
                        "           }                         " + "\n" +
                        "          delta_w[i] /=  Batch_Size;    " + "\n" +
                        "                                    " + "\n" +
                        "                                    " + "\n" +
                        "                                    " + "\n" +
                        "                                    " + "\n" +
                        "                                    " + "\n" +
                        "                                    " + "\n" +
                        "    }" + "\n" +
                        "}" + "\n";
        /*
                "extern \"C\"" + "\n" +
                        "__global__ void W_delta_Code(int Batch_Size, int len, int last_len ,int w_len ,double *delta, double *last_out, double *delta_w)" + "\n" +
                        "{" + "\n" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " + "\n" +
                        "    if ( i < w_len )                                 " + "\n" +
                        "    {                                              " + "\n" +
                        "        int j;                " + "\n" +
                        "        for( j=0; j<Batch_Size; j++){                 " + "\n" +
                        "            int  n = i / last_len;           " + "\n" +
                        "            int li = i % last_len;             " + "\n" +
                        "            int d = j * len + n;             " + "\n" +
                        "            delta_w[i] += delta[d] * last_out[j*last_len + li];  " + "\n" +
                        "         }                           " + "\n" +
                        "          delta_w[i]/=  Batch_Size;                       " + "\n" +
                        "                                    " + "\n" +
                        "    }" + "\n" +
                        "}" + "\n";
         */

        public final static String d_delta_Code =
                "extern \"C\"" + "\n" +
                        "__global__ void d_delta_Code(int Batch_Size, int len, double *delta, double *delta_d)" + "\n" +
                        "{" + "\n" +
                        "    int i = blockIdx.x * blockDim.x + threadIdx.x; " + "\n" +
                        "    if ( i < len )                                 " + "\n" +
                        "    {                                              " + "\n" +
                        "        delta_d[i]=0;                            " + "\n" +
                        "        int j;"      +  "\n" +
                        "        #pragma unroll 4                    " + "\n" +
                        "        for( j=0; j<Batch_Size; j++){            " + "\n" +
                        "            delta_d[i] -=  delta[j * len + i];     " + "\n" +
                        "        }                                       " + "\n" +
                        "        delta_d[i] /= Batch_Size;      " + "\n" +
                        "                           " + "\n" +
                        "                           " + "\n" +
                        "                           " + "\n" +
                        "    }" + "\n" +
                        "}" + "\n";
    }
}
