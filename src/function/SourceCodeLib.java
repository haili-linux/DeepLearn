package function;

public class SourceCodeLib {
    public final static String sigmoid_code_Name =  "sigmoid2323247730";
    public final static String sigmoid_code =
            "extern \"C\"" + "\n" +
            "__global__ void sigmoid2323247730(int n, double *in)" + "\n" +
            "{" + "\n" +
            "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
            "    if (i<n)" + "\n" +
            "    {" + "\n" +
            "        in[i] = 1.0 / ( 1.0 + exp(-in[i]) );" + "\n" +
            "    }" + "\n" +
            "}" + "\n";

    public final static String tanh_code_Name =  "tanh_2323247730";
    public final static String tanh_code =
            "extern \"C\"" + "\n" +
                    "__global__ void tanh_2323247730(int n, double *in)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if (i<n)" + "\n" +
                    "    {" + "\n" +
                    "        double x1 = exp(in[i]);" + "\n" +
                    "        double x2 = exp(-in[i]);" + "\n" +
                    "        in[i] = (x1 - x2) / (x1 + x2);" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";

    public final static String relu_code_Name =  "relu_2323247730";
    public final static String relu_code =
            "extern \"C\"" + "\n" +
                    "__global__ void relu_2323247730(int n, double *in)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if (i<n)" + "\n" +
                    "    {" + "\n" +
                    "        if( in[i] < 0 ) in[i] = 0;" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";

    public final static String Lrelu_code_Name =  "Lrelu_2323247730";
    public final static String Lrelu_code =
            "extern \"C\"" + "\n" +
                    "__global__ void Lrelu_2323247730(int n, double *in)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if (i<n)" + "\n" +
                    "    {" + "\n" +
                    "        if( in[i] < 0 ) in[i] = 0.001 * in[i];" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";






    public final static String tanh_d_code_Name =  "tanh_d_2323247730";
    public final static String tanh_d_code =
            "extern \"C\"" + "\n" +
                    "__global__ void tanh_d_2323247730(int n, double *nerve_out, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if (i<n)" + "\n" +
                    "    {" + "\n" +
                    "        out[i] *= 1 - nerve_out[i] * nerve_out[i];  " + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";

    public final static String sigmoid_d_code_Name =  "sigmoid_d_2323247730";
    public final static String sigmoid_d_code =
            "extern \"C\"" + "\n" +
                    "__global__ void sigmoid_d_2323247730(int n, double *nerve_out, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if (i<n)" + "\n" +
                    "    {" + "\n" +
                    "        out[i] *= nerve_out[i] * (1.0 - nerve_out[i]);" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";

    public final static String rule_d_code_Name =  "rule_d_2323247730";
    public final static String rule_d_code =
            "extern \"C\"" + "\n" +
                    "__global__ void rule_d_2323247730(int n, double *nerve_out, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if ( i < n )" + "\n" +
                    "    {" + "\n" +
                    "       if( nerve_out[i] < 0 )  out[i] = 0;" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";

    public final static String lrule_d_code_Name =  "lrule_d_2323247730";
    public final static String lrule_d_code =
            "extern \"C\"" + "\n" +
                    "__global__ void lrule_d_2323247730(int n, double *nerve_out, double *out)" + "\n" +
                    "{" + "\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
                    "    if ( i < n )" + "\n" +
                    "    {" + "\n" +
                    "       if( nerve_out[i] < 0 ) out[i] *= 0.01;" + "\n" +
                    "    }" + "\n" +
                    "}" + "\n";

}
