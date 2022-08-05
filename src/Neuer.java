import function.Fuction;
import function.Sigmoid;

import java.util.*;
import java.io.*;

class Neuer implements Cloneable,Serializable
{
	// 单个神经元
	public int input_n = 1;//输入变量数
	public double d = 0;//偏移量,.阈值
	public double lastIn = 0;//上一次输入
	public double lastOut = 0;//上一次输出
	public double delta = 0; //权值的梯度
	public double[] w_old;//上一次权的值
	public double[] w;//权值

	public Fuction ACT_function; // 激活函数
	public double[] data_list;//一个储存空间,和逻辑无关，灵活运用
	public double data = 0;//一个储存空间,和逻辑无关，灵活运用
	
	//默认输入维度为1
	public Neuer(){
		w = new double[input_n];
		d = Math.random()*2 - 1;//偏置值随机
		init();
		ACT_function = new Sigmoid();
		data_list = new double[input_n];
	}

	//指定输入维度
    public Neuer(int n){
		input_n = n;
		w = new double[n];
		d = Math.random()*2 - 1;
		init();
		ACT_function = new Sigmoid();
		data_list = new double[input_n];
	}

	public Neuer(int n,Fuction act_function){
		input_n = n;
		w = new double[n];
		d = /*(act_function.id==3) ? -0.8:*/Math.random()*2-1;
		init();
		ACT_function = act_function;
		data_list = new double[input_n];
	}

	//指定输入维度和偏置值
	public Neuer(int n,double d,Fuction act_funtion){
		input_n = n;
		w = new double[n];
		this.d = d;
		init();
		ACT_function = act_funtion;
		data_list = new double[input_n];
	}

	//初始化
	private void init(){
		w_old = w;
		double a = 0;
		for(int i=0;i<w.length;i++){
			//初始权值随机
			w[i] = Math.random();
			a += w[i];
		}
		for(int i=0;i<w.length;i++)
			w[i] /= a;
	}
   
	public void setW(int n,double newW){
		w_old[n] = w[n];//同时修改w_old
		w[n] = newW;
	}
	
	//激活函数
	public double act_f(double x){
		return ACT_function.f(x);
    }

	//严格定义in[],不然出错,结果记忆
	public double out(double[] in){
		double x = 0;
		for(int i= 0;i<input_n;i++)
			x += w[i] * in[i];
		x += -d;
		lastIn = x;
		return  lastOut = act_f(x);
	}

	//不记忆结果
	public double out_notSave(double[] in){
		double x = 0;
		for(int i= 0;i<input_n;i++)
			x += w[i] * in[i];
		x += -d;
		return  act_f(x);
	}

	public double LastIn_notSave(double[] in){
		double x = 0;
		for(int i= 0;i<input_n;i++)
			x += w[i] * in[i];
		x += -d;
		return  x;
	}

	//扩展输入维度, n:要扩展的维度数,默认加在最后
	public void addInput_n(int add_number){
		if(add_number>0){
			input_n += add_number;
			double[] new_W = new double[input_n];
			double[] new_data_list = new double[input_n];
			
			for(int i=0;i<w.length;i++){
				new_W[i] = w[i];
				new_data_list[i] = data_list[i];
			}
			double a = 0;
			for(int i=w.length;i<input_n;i++){
				new_W[i] = Math.random();
				a += new_W[i];
		    }
			
			if(a>1)//防止过饱和
			  for(int i=w.length;i<input_n;i++)
			     new_W[i]/=a;
			
			w = new_W;
			w_old = w;
			data_list = new_data_list;
		}
	}

	//减少输入维度, n:要扩展的维度数,默认加在最后
	public void deleteInput_n(int de_number){
		if(de_number>0){
			input_n -= de_number;
			w = Arrays.copyOf(w,input_n);
			w_old = w;
			data_list = Arrays.copyOf(data_list,input_n);
		}
	}
	
	//设置输入维度
	public void newInput_n(int n){
		if(n>0){
		   input_n = n;
	       w = arraysOpon(w,new double[n]);
		   w_old = w;
		   data_list = new double[n];
	    }
	}
	
	//数组映射
	public double[] arraysOpon(double[] a,double[] t){
		int al = a.length;
		int tl = t.length;
		if(al==tl) return a;
		double dindex;
		double index = 0;
		if(al>tl){
			dindex = (double)al/tl;
			for(int i=0;i<tl;i++){
				int j = doubleToInt(index);
				if(j>=al)
					t[i] = a[al-1];
				else
				    t[i] = a[j];
				index += dindex;
			}
		}else{
			dindex = (double)tl/al;
			for(int i=0;i<al;i++){
				int j = doubleToInt(index);
				if(j>=tl)
					t[tl-1] = a[i];
				else
					t[j] = a[i];
				index += dindex;
			}
		}
		return t;
	}

	//浮点数转整形，四舍五入
	final private int doubleToInt(double x){
		double a = x%1;
		int r = (int)x;
		if(a>=0.5) r++;
		return r;
	}
	
	@Override
	public String toString() {
		// TODO: Implement this method
		return "w:" + Arrays.toString(w) + "  d:"+d;
	}

	@Override
    protected Object clone()  {
        Neuer p =null;
        try {
            p= (Neuer)super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return p;
    }
	public Object deepClone() throws IOException, ClassNotFoundException{
		ByteArrayOutputStream bo = new ByteArrayOutputStream();
		ObjectOutputStream os = new ObjectOutputStream(bo);
		os.writeObject(this);

		ByteArrayInputStream bi = new ByteArrayInputStream(bo.toByteArray());
		ObjectInputStream is = new ObjectInputStream(bi);
		return is.readObject();
	}
}
