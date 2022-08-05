import DeltaOptimizer.BaseOptimizer;
import DeltaOptimizer.BaseOptimizerInterface;
import function.*;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class BpNetwork implements Cloneable,Serializable
{
	public int input_n;//输入维度
	public int output_n;//输出维度
	public double nl;//学习率 0～1
	public int act_fuctiom_ID;//网络的激活函数
	public int[] hid_n; //隐藏层神经元结构


	public double dError; //误差
	public Fuction Loss_function;

	//public double[] inNeuer_out;//输入层每个神经元的输出
	public double[] outNeuer_out;//输出
	public double[][] hiddenNeuer_out;//隐藏层每个输出

	//public Neuer[] input_Neuer;//输入层神经元
	public Neuer[] output_Neuer;//输出层神精元
	public Neuer[][] hidden_Neuer;/*第i层*//*i层神经元数量*///隐藏层神经元

	public String EXPLAIN; //神经网络说明

	ExecutorService upThreadPool;//用于并行计算的线程池

	BaseOptimizerInterface deltaOptimizer; //梯度优化器

	//in:输入量维度, outn:输出结果维度, ng:学习效率  hidden_:隐藏层每层神经元数量
	public BpNetwork(int in_vector, int out_vector, double ng, Fuction act_fuction , int[] hidden_){
		input_n = in_vector;
		output_n = out_vector;
		nl = ng;
		hid_n = hidden_;
		act_fuctiom_ID = act_fuction.id;

		init(act_fuction);
		upThreadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		deltaOptimizer = new BaseOptimizer();
	}

	//从文件初始化
	public BpNetwork(String file){
		readFile(file);
		System.out.println("说明:"+EXPLAIN);
		System.out.println("输入维度: " + input_n);
		System.out.println("输出维度: " + output_n);
		System.out.println("学习率: " + nl);
		System.out.println("激活函数: " + getFuctionById(act_fuctiom_ID).getClass().toString());
		System.out.println("隐藏层结构: " + Arrays.toString(hid_n));
		deltaOptimizer = new BaseOptimizer();
		upThreadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
	}
	
	//计算输出
	public double[] out_(double[] in){
		if(in.length!=input_n){ 
		    System.out.println("输入数据格式错误");
			return null;
		}
		if(_in_max>1)//归一化
		   for(int i=0;i<in.length;i++)
		       in[i] /= _in_max;
			   
		double[] r = out(in);
		
		if(_out_max>1)
		  for(int i=0;i<r.length;i++)
		    r[i] *= _out_max;
		
		return r;
	}


	//训练神经,fix(输入，目标，batch_size,训练次数)
	public double fit(double[][] in,double[][] t,int batch_size,int n){
		//参数检查
		if(batch_size<1 &&n <0) return 99999;
		
		if(batch_size==1)//batch_size=1,随机梯度下降
			for(int i=0;i<n;i++)
			   for(int j=0;j<in.length;j++)
			            upgrade(in[j],t[j]);
		else if(batch_size >= in.length)//batch_size和训练集一样，全批量梯度下降
			for(int i=0;i<n;i++) upgrade_batch(in,t,getThread_n(batch_size));
		else//mini-batch
		    for(int i=0;i<n;i++) upgrade_mini_batch(in,t,batch_size,getThread_n(batch_size));
		
		//计算误差
		double error = 0;
		for(int i=0;i<in.length;i++)
			error += Error(out(in[i]),t[i]);
		
		return dError = error/in.length;
	}
	public double fit(double[][] in,double[][] t,int batch_size,int n,int Thread_n){
		//参数检查
		if(batch_size<1 &&n <0) return 99999;
		//获取cpu核心数
		int core_number = Runtime.getRuntime().availableProcessors();
		if(Thread_n>core_number) Thread_n = core_number;

		if(batch_size==1)//batch_size=1,随机梯度下降
			for(int i=0;i<n;i++)
				for(int j=0;j<in.length;j++)
					upgrade(in[j],t[j]);
		else if(batch_size >= in.length)//batch_size和训练集一样，全批量梯度下降
			for(int i=0;i<n;i++){
				upgrade_batch(in,t,Thread_n);
			}
		else//mini-batch
		    for(int i=0;i<n;i++) upgrade_mini_batch(in,t,batch_size,Thread_n);

		//计算误差
		double error = 0;
		for(int i=0;i<in.length;i++)
			error += Error(out(in[i]),t[i]);

		return dError = error/in.length;
	}
	public double fit_time(double[][] in,double[][] t,int batch_size,int Thread_n,int time_min){
		//参数检查
		if(batch_size<1 && time_min <0) return 99999;
		//获取cpu核心数
		int core_number = Runtime.getRuntime().availableProcessors();
		if(Thread_n>core_number) Thread_n = core_number;

		long startTime = System.currentTimeMillis();
		if(batch_size==1)//batch_size=1,随机梯度下降
		{
			while (true) {
				for (int j = 0; j < in.length; j++) upgrade(in[j], t[j]);
				long Time = (System.currentTimeMillis() - startTime) / 60000;
				if(Time >= time_min) break;
			}
		}else if(batch_size >= in.length)//batch_size和训练集一样，全批量梯度下降
		{
			while (true) {
				upgrade_batch(in, t, Thread_n);
				long Time = (System.currentTimeMillis() - startTime) / 60000;
				if(Time >= time_min) break;
			}
		}
		else//mini-batch
		{
			while (true) {
				upgrade_mini_batch(in, t, batch_size, Thread_n);
				long Time = (System.currentTimeMillis() - startTime) / 60000;
				if(Time >= time_min) break;
			}
		}
		//计算误差
		double error = 0;
		for(int i=0;i<in.length;i++)
			error += Error(out(in[i]),t[i]);

		return dError = error/in.length;
	}

	//测试一个数据集上的误差
	public double test_(double[][] in,double[][] t){
		double error = 0;
		for(int i=0;i<in.length;i++)
			error += Error(out(in[i]),t[i]);
		return dError = error/in.length;
	}
	
	//数据归一化
	double _in_max=1.0,_out_max=1.0;//选出训练集中最大值
	public double[][][] data_of_one(double[][] input,double[][] output){
		_in_max = getListMax(input);
		_out_max = getListMax(output);
	    if(_in_max>1)
			for(int i=0;i<input.length;i++)
				for(int j=0;j<input[i].length;j++)
					input[i][j] = input[i][j]/_in_max;
		else _in_max = 1;
		if(_out_max>1)
			for(int i=0;i<output.length;i++)
				for(int j=0;j<output[i].length;j++)
					output[i][j] = output[i][j]/_out_max;
		else _out_max = 1;
		return new double[][][]{input,output};
	}

	 
	//增加输入维度, n:要扩展的维度数,默认加在最后
	/*public void addInput_n(int add_number,Fuction act_fuction){
		if(add_number>0){
		   input_n += add_number;
		   ArrayList<Neuer> inputNeuer =new ArrayList<Neuer>(Arrays.asList(input_Neuer));
		   input_Neuer = new Neuer[input_n];
		   inNeuer_out = new double[input_n];
		   int i;
		   //复制
		   for(i=0;i<inputNeuer.size();i++)
			   input_Neuer[i] = inputNeuer.get(i);
		   //新增
		   for(i=inputNeuer.size();i<input_n;i++)
		       input_Neuer[i] = new Neuer(1,act_fuction);
			 
		   //调整下一层神经元输入数
		   for(i=0;i<hidden_Neuer[0].length;i++)
			   hidden_Neuer[0][i].addInput_n(add_number);
		   
		}
	}

	public void addInput_n(int add_number){
		addInput_n(add_number,getFuctionById(act_fuctiom_ID));
	}

	//减少输入维度, n:要减的维度数,默认加在最后
	public void deleteInput_n(int de_number){
		if(de_number>0 && input_n>de_number){
		     input_n -= de_number;
		     input_Neuer = Arrays.copyOf(input_Neuer,input_n);
		     inNeuer_out = new double[input_n];
		  
		     //下一层
		     for(int i=0;i<hidden_Neuer[0].length;i++)
			     hidden_Neuer[0][i].deleteInput_n(de_number);
		}
	}*/
	
	//扩展输出维度,默认在最后
	public void addOutput_n(int add_number,Fuction act_fuction){
		if(add_number>0){
			output_n += add_number;
			Neuer[] newOutput_Neuer = new Neuer[output_n];
			
			int i;
			for(i=0;i<output_Neuer.length;i++)
			     newOutput_Neuer[i] = output_Neuer[i];
			
			for(i=output_Neuer.length;i<output_n;i++)
			     newOutput_Neuer[i] = new Neuer(hidden_Neuer[hidden_Neuer.length-1].length,act_fuction);
			
			output_Neuer = newOutput_Neuer;
			outNeuer_out = new double[output_n];	
		}
	}
	public void addOutput_n(int add_number){
		addOutput_n(add_number,getFuctionById(act_fuctiom_ID));
	}

	//减少输出维度, n:要减的维度数,默认加在最后
	public void deleteOutput_n(int de_number){
		if(de_number>0 && output_n>de_number){
			output_n -= de_number;
			outNeuer_out = new double[output_n];
			output_Neuer = Arrays.copyOf(output_Neuer,output_n);
		}
	}
	
	//在第n隐藏层上增加神经元
	public void addNeuerInHidden(int n,int add_number,Fuction act_fuction){
		if(n<hidden_Neuer.length && n>=0 && add_number>0){
			hid_n[n] += add_number;
			Neuer[] newlist = new Neuer[hid_n[n]];
			
			int i;
			for(i=0;i<hidden_Neuer[n].length;i++)
				newlist[i] = hidden_Neuer[n][i];
			
			for(i=hidden_Neuer[n].length; i<hid_n[n]; i++)
			    newlist[i] = new Neuer(hidden_Neuer[n][0].w.length,act_fuction);
			
			//增加
			hidden_Neuer[n] = newlist;
			hiddenNeuer_out[n] = new double[hid_n[n]];
				
			//下一层
			if(n==hidden_Neuer.length-1)
				for(i=0;i<output_n;i++) output_Neuer[i].addInput_n(add_number);
			else
				for(i=0;i<hidden_Neuer[n+1].length;i++)
				     hidden_Neuer[n+1][i].addInput_n(add_number);
		}
	}
	public void addNeuerInHidden(int n,int add_number){
		addNeuerInHidden(n,add_number,getFuctionById(act_fuctiom_ID));
	}

	//在第n隐藏层上减少神经元
	public void deleteNeuerInHidden(int n,int de_number){
		if(de_number>0 && n<hidden_Neuer.length && n>=0 && hid_n[n]>de_number ){
			hid_n[n] -= de_number;
			hidden_Neuer[n] = Arrays.copyOf(hidden_Neuer[n],hid_n[n]);
			hiddenNeuer_out[n] = new double[hid_n[n]];
			
			//下一层
			if(n==hidden_Neuer.length-1)
				for(int i=0;i<output_n;i++) output_Neuer[i].deleteInput_n(de_number);
			else
				for(int i=0;i<hidden_Neuer[n+1].length;i++)
					hidden_Neuer[n+1][i].deleteInput_n(de_number);		
		}
	}
	
	//在第n层前插入1层含有k个神经元的隐藏层
	public void addHiddenNeuer(int n,int k,Fuction act_fuction){
		if(n>=0 && k>0){
			Neuer[] newN = new Neuer[k];
			boolean flag = false;
			//初始化新加入的层
			if(n==0){
				for(int i=0;i<k;i++) newN[i] = new Neuer(input_n,act_fuction);
			}else if(flag = n>=hidden_Neuer.length){
				for(int i=0;i<k;i++) newN[i] = new Neuer(hid_n[hid_n.length-1],act_fuction);
			}else{
				for(int i=0;i<k;i++) newN[i] = new Neuer(hidden_Neuer[n-1].length,act_fuction);
			}
			
			//插入新层
			ArrayList<Neuer[]> newHiddenN = new ArrayList(Arrays.asList(hidden_Neuer));
			if(n<newHiddenN.size())
			    newHiddenN.add(n,newN);
			else
				newHiddenN.add(newN);

			int[] newHid_n = new int[newHiddenN.size()];
			double[][] newHidden_Out = new double[newHiddenN.size()][];
			for(int i=0;i<newHid_n.length;i++){
				newHid_n[i] = newHiddenN.get(i).length;
				newHidden_Out[i] = new double[newHid_n[i]];
			}
			
			//替换
			hidden_Neuer = newHiddenN.toArray(new Neuer[newHiddenN.size()][]);
			hiddenNeuer_out = newHidden_Out;
			hid_n = newHid_n;
			
			//修改下一层
			if(flag)
				for(int i=0;i<output_Neuer.length;i++){
					output_Neuer[i].newInput_n(k);
				}
			else
				for(int i=0;i<hidden_Neuer[n+1].length;i++){
					hidden_Neuer[n+1][i].newInput_n(k);
				}
				
		   System.out.println(Arrays.toString(hidden_Neuer[1]));
		}
	}
	public void addHiddenNeuer(int n,int k){
		addHiddenNeuer(n,k,getFuctionById(act_fuctiom_ID));
	}

	//删除第n层神经元
	public void deletcHiddenNeuer(int n){
		if(n>=0 && n<hidden_Neuer.length && hidden_Neuer.length>1){
			ArrayList<Neuer[]> newHiddenN = new ArrayList(Arrays.asList(hidden_Neuer)); 
			//删除
			newHiddenN.remove(n);
			
			int[] newHid_n = new int[newHiddenN.size()];
			double[][] newHidden_Out = new double[newHiddenN.size()][];
			for(int i=0;i<newHid_n.length;i++){
				newHid_n[i] = newHiddenN.get(i).length;
				newHidden_Out[i] = new double[newHid_n[i]];
			}
			
			//替换
			hidden_Neuer = newHiddenN.toArray(new Neuer[newHiddenN.size()][]);
			hiddenNeuer_out = newHidden_Out;
			hid_n = newHid_n;
			
			//修改下一层
			if(n==0){
				for(int i=0;i<hid_n[0];i++)
					hidden_Neuer[0][i].newInput_n(input_n);
			}else if(n>=hid_n.length){
				for(int i=0;i<output_Neuer.length;i++)
					output_Neuer[i].newInput_n(hid_n[hid_n.length-1]);
			}else{
				for(int i=0;i<hid_n[n];i++)
				    hidden_Neuer[n][i].newInput_n(hid_n[n-1]);
			}
		}
	}

    //把神经网络保存到文件
	public void saveInFile(String path) {
		File f = new File(path);
		if (f.exists()) {
			String p1 = path.substring(0, path.lastIndexOf("."));
			for (int i = 0; ; i++) {
				f = new File(path = p1 + "_" + i + ".log");
				if (!f.exists()) break;
			}
		}
		try {
			f.createNewFile();
		} catch (Exception e) {
		}

		if(f.isFile()) {
			FileWriter fw = null;
			try {
				fw = new FileWriter(f, true);
			} catch (IOException e) {
				e.printStackTrace();
			}
			PrintWriter pw = new PrintWriter(fw);

			pw.println("explain:" + EXPLAIN);
			pw.println(SInt("in_vector", input_n));
			pw.println(SInt("out_vector", output_n));
			pw.println(Sdouble("nl", nl));
			pw.println(sIntArrays("hid_n", hid_n));
			pw.println(Sdouble("de", dError));
			pw.println(Sdouble("_in_max", _in_max));
			pw.println(Sdouble("_out_max", _out_max));
			pw.println(SInt("ACT_FUCTION",act_fuctiom_ID));
			pw.println(SInt("LOSS_FUCTION", Loss_function.id));


			//输出层
			for (int i = 0; i < output_n; i++) {
				String name = "outputNeuer[" + i + "].";
				pw.println(SInt(name + "act_fuction_id",output_Neuer[i].ACT_function.id));
				pw.println(Sdouble(name + "d", output_Neuer[i].d));
				for (int j = 0; j < output_Neuer[i].w.length; j++)
					pw.println(Sdouble(name + "w" + j, output_Neuer[i].w[j]));
			}
			//隐藏层
			for (int n = 0; n < hidden_Neuer.length; n++)//第n层
				for (int i = 0; i < hidden_Neuer[n].length; i++) {//第i个
					String name = "hiddenNeuer[" + n + "][" + i + "].";
					pw.println(SInt(name + "act_fuction_id",hidden_Neuer[n][i].ACT_function.id));
					pw.println(Sdouble(name + "d", hidden_Neuer[n][i].d));
					for (int j = 0; j < hidden_Neuer[n][i].w.length; j++)
						pw.println(Sdouble(name + "w" + j, hidden_Neuer[n][i].w[j]));
				}

			pw.flush();
			try {
				fw.flush();
				pw.close();
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	//对象深复制
	@Override
    public Object clone() throws CloneNotSupportedException {
        BpNetwork p = (BpNetwork)super.clone();
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

	/****
	*****
	以下是私有函数
	*****
	*****/
	
	//神经网络输出
	private double[] out(double[] in){
		//计算输入层的输出
		int i;

		for(i=0; i < hidden_Neuer[0].length; i++){
		 	hiddenNeuer_out[0][i] = hidden_Neuer[0][i].out(in);
		}

		//计算隐藏层输出
		for( i = 1; i < hidden_Neuer.length; i++){
			for( int j = 0; j < hidden_Neuer[i].length; j++ )
				hiddenNeuer_out[i][j] = hidden_Neuer[i][j].out(hiddenNeuer_out[i-1]);
		}

		//计算最终输出输
		for(i=0;i<outNeuer_out.length;i++)
			outNeuer_out[i] = output_Neuer[i].out(hiddenNeuer_out[hiddenNeuer_out.length-1]);
		return outNeuer_out;
	}

	//动态选择线程数量
	private int getThread_n(int batch)  {
		double netSize = 0; //网络规模
		for(int i=1;i<hid_n.length;i++)
			netSize += hid_n[i-1]*hid_n[i];
		
		int n = (int) ((netSize*batch)/(1e5));
		
		//获取cpu核心数
		int core_number = Runtime.getRuntime().availableProcessors();
		
		if(n>core_number) n = core_number;
		
		return n;
	}


	//批量梯度下降更新thread_n:开启的线程数
	private void upgrade_batch(double[][] in,double[][] t,int Thread_n){
		int len = in.length;
		int i;
		if(Thread_n<2){
			//单线程
		    for(int j=0;j<len;j++) {
			 	upgradeBatch(in[j], t[j]);//耗时点
			}
		}else{
		    //多线程运算
		    Thread_upgrade upthread = new Thread_upgrade(in,t);
		    Future[] futureList = new Future[Thread_n];
		    for(i=0;i<Thread_n;i++)
		    	futureList[i] = upThreadPool.submit(new Thread(upthread));

			try {
				for(Future future: futureList) future.get();
			}catch (InterruptedException e){
				e.printStackTrace();
			}catch(ExecutionException e){
				e.printStackTrace();
			}
		}


		for (i = 0; i < output_Neuer.length; i++) {//star for
				for (int j = 0; j < output_Neuer[i].w.length; j++) {
					//优化器
					output_Neuer[i].data_list[j] = deltaOptimizer.DELTA(output_Neuer[i].data_list[j] / len);

					output_Neuer[i].setW(j, output_Neuer[i].w[j] + nl * output_Neuer[i].data_list[j]);
					output_Neuer[i].data_list[j] = 0;
				}



			    output_Neuer[i].data = deltaOptimizer.DELTA(output_Neuer[i].data / len);


				output_Neuer[i].d += nl * output_Neuer[i].data;
				output_Neuer[i].data = 0;
		}//end for


		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--)
			for (int j = 0; j < hidden_Neuer[i].length; j++) {
				for (int k = 0; k < hidden_Neuer[i][j].w.length; k++) {

					    hidden_Neuer[i][j].data_list[k] = deltaOptimizer.DELTA(hidden_Neuer[i][j].data_list[k] / len);

						hidden_Neuer[i][j].setW(k, hidden_Neuer[i][j].w[k] + nl * hidden_Neuer[i][j].data_list[k]);
						hidden_Neuer[i][j].data_list[k] = 0;
				}//end for


				hidden_Neuer[i][j].data = deltaOptimizer.DELTA(hidden_Neuer[i][j].data /len);

				hidden_Neuer[i][j].d += nl * hidden_Neuer[i][j].data;
				hidden_Neuer[i][j].data = 0;
			}//end for


	    if(hidden_Neuer.length>1)//和输入层连接的隐藏层
	    	for (i = 0; i < hidden_Neuer[0].length; i++) {
	    		for (int j = 0; j < hidden_Neuer[0][i].w.length; j++) {//更新

					hidden_Neuer[0][i].data_list[j] = deltaOptimizer.DELTA(hidden_Neuer[0][i].data_list[j] / len);

	    			hidden_Neuer[0][i].setW(j, hidden_Neuer[0][i].w[j] + nl * hidden_Neuer[0][i].data_list[j] );
	    			hidden_Neuer[0][i].data_list[j] = 0;
	    		}

				hidden_Neuer[0][i].data = deltaOptimizer.DELTA(hidden_Neuer[0][i].data / len);

	    		hidden_Neuer[0][i].d += nl * hidden_Neuer[0][i].data;
	    		hidden_Neuer[0][i].data = 0;
	    	}//end for

	}
	
	//mini-batch
	private void upgrade_mini_batch(double[][] in,double[][] t,int batch_size,int thread_n){
		double[][] in_batch = new double[batch_size][];
		double[][] t_batch = new double[batch_size][];
		for(int j=0;j<in_batch.length;j++){
			//随机选出n个作为mini-batch
			int index = (int)(Math.random()*in.length);
			in_batch[j] = in[index];
			t_batch[j] = t[index];
		}
		upgrade_batch(in_batch,t_batch,thread_n);
	}

	//随机梯度下降更新，batch-size = 1;
	public void upgrade(double[] input,double[] target){
		double[] out = out(input);
		int i;
		//输出层
		for(i=0;i<output_Neuer.length;i++){//star for

			if(output_Neuer[i].ACT_function.id==1 && Loss_function.id==11)
				 output_Neuer[i].delta = -(out[i] - target[i]);
			else
			     output_Neuer[i].delta = -Loss_function.f_derivate(out[i],target[i]) * output_Neuer[i].ACT_function.f_derivate(output_Neuer[i].lastOut);

			double x1 = output_Neuer[i].delta;

 			for(int j=0;j<output_Neuer[i].w.length;j++) {

				output_Neuer[i].setW(j, output_Neuer[i].w[j] + nl * deltaOptimizer.DELTA(x1 * hiddenNeuer_out[hidden_Neuer.length - 1][j]));//= tnl * deltaOptimizer.DELTA( x1 * hiddenNeuer_out[hidden_Neuer.length-1][j]) );

			}

			output_Neuer[i].d += nl * deltaOptimizer.DELTA(-x1);
		}//end for

		double delta;
		int n;
		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--){
			for(int j=0;j<hidden_Neuer[i].length;j++){
				delta = 0;
				if(i == hidden_Neuer.length-1){
					n = output_Neuer.length;
					for(int k=0; k<n; k++)
						delta += output_Neuer[k].delta * output_Neuer[k].w_old[j];
				}
				else{
					n = hidden_Neuer[i+1].length;
					for(int k=0; k<n; k++)
						delta += hidden_Neuer[i+1][k].delta * hidden_Neuer[i+1][k].w_old[j];
				}// end if

				delta *= hidden_Neuer[i][j].ACT_function.f_derivate(hidden_Neuer[i][j].lastOut);

				//更新权值和阀值
				for (int k = 0; k < hidden_Neuer[i][j].w.length; k++)
					hidden_Neuer[i][j].setW(k,hidden_Neuer[i][j].w[k] + nl * deltaOptimizer.DELTA(delta * hiddenNeuer_out[i - 1][k]));

				hidden_Neuer[i][j].d += nl * deltaOptimizer.DELTA(-delta);

			}


		}//end for


		if(hidden_Neuer.length>1)//和输入层连接的隐藏层
			for(i=0;i<hidden_Neuer[0].length;i++){
				 delta = 0;
				for(int j=0;j<hidden_Neuer[1].length;j++)
					delta += hidden_Neuer[1][j].delta * hidden_Neuer[1][j].w_old[i];

				delta *= hidden_Neuer[0][i].ACT_function.f_derivate(hidden_Neuer[0][i].lastOut);
				hidden_Neuer[0][i].delta = delta;

				for(int j=0;j<hidden_Neuer[0][i].w.length;j++)//更新
					hidden_Neuer[0][i].setW(j,hidden_Neuer[0][i].w[j] + nl * deltaOptimizer.DELTA(delta * input[j]));

				hidden_Neuer[0][i].d += nl * deltaOptimizer.DELTA(-delta);
			}//end for

	}// end upgrade

	//批量梯度下降，多线程并行计算
	private class Thread_upgrade implements Runnable{
		private boolean finish = false;
		private double[][] in;
		private double[][] t;
		private boolean[] nfinish;//是否计算完成
		private boolean[] lock;//是否锁了(正在计算)
		
		public Thread_upgrade(double[][] in_,double[][] t_){
			in = in_;
			t = t_;
			nfinish = new boolean[in.length];
			lock = new boolean[in.length];
		}
		
		public void run(){
			upThreadData td = new upThreadData();
			while(true){
				if(finish) break;
				boolean flag = true;
				for(int p=0;p<in.length;p++)
					if(!nfinish[p]){	//第i个未完成
						flag = false; 
						if(!lock[p]){//i个未锁
							lock[p] = true; //上锁
						    //doing
							upgrade_inThread(in[p],t[p],td);
							//处理完成,
							nfinish[p] = true;
						}
				    }	  
				if(flag){
					finish = true;//已完成
					break;
				}
			}//end while
		}//end run
	}
	private class upThreadData{//数据复用

		double[] out = new double[outNeuer_out.length];
		double[] outNeuer_delta = new double[output_Neuer.length];

		double[][] hiddenNeuer_out_ = new double[hidden_Neuer.length][];//隐藏层每个输出
		double[][] hiddenNeuer_delta = new double[hidden_Neuer.length][];//隐藏层每个delta
        
		public upThreadData(){
		    for(int i=0;i<hid_n.length;i++){//for
			   hiddenNeuer_out_[i] = new double[hid_n[i]];
			   hiddenNeuer_delta[i] = new double[hid_n[i]];
		  }
		}// end for
	}
	private void upgrade_inThread(double[] in,double[] t, upThreadData td){
		int i;
		for(int j=0;j<hidden_Neuer[0].length;j++)
				td.hiddenNeuer_out_[0][j] = hidden_Neuer[0][j].act_f(hidden_Neuer[0][j].LastIn_notSave(in));


		//计算隐藏层输出
		for( i = 1; i < hidden_Neuer.length; i++ )
			for(int j=0;j<hidden_Neuer[i].length;j++)
				td.hiddenNeuer_out_[i][j] = hidden_Neuer[i][j].act_f(hidden_Neuer[i][j].LastIn_notSave(td.hiddenNeuer_out_[i-1]));



		//计算最终输出输
		for( i = 0; i < outNeuer_out.length; i++ )
			td.out[i] = output_Neuer[i].act_f(output_Neuer[i].LastIn_notSave(td.hiddenNeuer_out_[td.hiddenNeuer_out_.length-1]));



		double delta;

		//以下计算梯度
		for(i=0;i<output_Neuer.length;i++){//star for
			if(output_Neuer[i].ACT_function.id==1 && Loss_function.id==11)
				 delta = -(td.out[i] - t[i]);
			else
			     delta = -Loss_function.f_derivate(td.out[i],t[i]) * output_Neuer[i].ACT_function.f_derivate(td.out[i]);

			td.outNeuer_delta[i] = delta;
			for(int j=0;j<output_Neuer[i].w.length;j++)
				output_Neuer[i].data_list[j] += delta * td.hiddenNeuer_out_[hidden_Neuer.length-1][j];

			output_Neuer[i].data += -delta;
		}//end for

		int n;
		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--){
			for(int j=0;j<hidden_Neuer[i].length;j++){
				delta = 0;
				if(i == hidden_Neuer.length-1){
					n = output_Neuer.length;
					for(int k=0; k<n; k++)
						delta += td.outNeuer_delta[k] * output_Neuer[k].w[j];//w_old[j];
				}
				else{
					n = hidden_Neuer[i+1].length;
					for(int k=0; k<n; k++)
						delta += td.hiddenNeuer_delta[i+1][k] * hidden_Neuer[i+1][k].w[j];//w_old[j];
				}// end if
				delta *= hidden_Neuer[i][j].ACT_function.f_derivate(td.hiddenNeuer_out_[i][j]);
				td.hiddenNeuer_delta[i][j] = delta;
				//计算更新权值和阀值0
				for(int k=0; k<hidden_Neuer[i][j].w.length;k++)
					hidden_Neuer[i][j].data_list[k] += delta * td.hiddenNeuer_out_[i-1][k];

				hidden_Neuer[i][j].data += -delta;
			}
		}//end for

		if(hidden_Neuer.length>1)//和输入层连接的隐藏层
			for(i=0;i<hidden_Neuer[0].length;i++){
				delta = 0;
				for(int j=0;j<hidden_Neuer[1].length;j++)
					delta += td.hiddenNeuer_delta[1][j] * hidden_Neuer[1][j].w[i];//w_old[i];

				delta *= hidden_Neuer[0][i].ACT_function.f_derivate(td.hiddenNeuer_out_[0][i]);
				td.hiddenNeuer_delta[0][i] = delta;

				for(int j=0;j<hidden_Neuer[0][i].w.length;j++)//更新
					hidden_Neuer[0][i].data_list[j] += delta * in[j];

				hidden_Neuer[0][i].data += -delta;
			}//end for

	}



	//batch-size>1	批量梯度下降，单线程
	private double upgradeBatch(double[] input, double[] target){
		//System.out.println("批量梯度下降，单线程");
		double[] out = out(input);
		//System.out.println("out " + Arrays.toString(out));
		int i;
		//输出层
		for(i=0;i<output_Neuer.length;i++){//star for

			if(output_Neuer[i].ACT_function.id==1 && Loss_function.id==11)
				output_Neuer[i].delta = -(out[i] - target[i]);
			else
				output_Neuer[i].delta = -Loss_function.f_derivate(out[i],target[i]) * output_Neuer[i].ACT_function.f_derivate(output_Neuer[i].lastOut);

			double x1 = output_Neuer[i].delta;//mmmmmmmm
			//System.out.print("  hd2: " + -Loss_function.f_derivate(out[i],target[i]));
			for(int j=0;j<output_Neuer[i].w.length;j++)
				output_Neuer[i].data_list[j] += x1 * hiddenNeuer_out[hidden_Neuer.length-1][j];

			output_Neuer[i].data += -x1;
		}//end for

		double delta;
		//隐藏层
		for(i=hidden_Neuer.length-1;i>0;i--){
			for(int j=0;j<hidden_Neuer[i].length;j++){
			    delta = 0;
				int n;
				if(i == hidden_Neuer.length-1){
					n = output_Neuer.length;
				    for(int k=0; k<n; k++)
						delta += output_Neuer[k].delta * output_Neuer[k].w[j];
				}
				else{
					n = hidden_Neuer[i+1].length;
				    for(int k=0; k<n; k++)
					    delta += hidden_Neuer[i+1][k].delta * hidden_Neuer[i+1][k].w[j];
				}// end if
				delta *= hidden_Neuer[i][j].ACT_function.f_derivate(hidden_Neuer[i][j].lastOut);
				hidden_Neuer[i][j].delta = delta;


				//计算更新权值和阀值
				for(int k=0; k<hidden_Neuer[i][j].w.length;k++)
					hidden_Neuer[i][j].data_list[k] += delta * hiddenNeuer_out[i-1][k];

				hidden_Neuer[i][j].data += -delta;
			}
		}//end for

	    if(hidden_Neuer.length>1)//和输入层连接的隐藏层
			for(i=0;i<hidden_Neuer[0].length;i++){
				delta = 0;

				for(int j=0;j<hidden_Neuer[1].length;j++)
					delta += hidden_Neuer[1][j].delta * hidden_Neuer[1][j].w[i];

				delta *= hidden_Neuer[0][i].ACT_function.f_derivate(hidden_Neuer[0][i].lastOut);
				hidden_Neuer[0][i].delta = delta;

				for(int j=0;j<hidden_Neuer[0][i].w.length;j++)//更新
					hidden_Neuer[0][i].data_list[j] += delta * input[j];

				hidden_Neuer[0][i].data += -delta;
			}//end for

		return Error(out,target);
	}// end upgrade

	
	//初始化
	private void init(Fuction act_fuction){
		Loss_function = new MSELoss();

		output_Neuer = new Neuer[output_n];
		outNeuer_out = new double[output_n];
		for(int i=0;i<output_Neuer.length;i++)
			output_Neuer[i] = new Neuer(hid_n[hid_n.length-1],act_fuction);

		hidden_Neuer = new Neuer[hid_n.length][];
		hiddenNeuer_out = new double[hid_n.length][];
		for(int i=0;i<hid_n.length;i++){//for
			Neuer n[]=new Neuer[hid_n[i]];
			hiddenNeuer_out[i] = new double[hid_n[i]];
			if(i==0)
				for(int j=0;j<n.length;j++) 
					n[j] = new Neuer(input_n,act_fuction);
			else 
				for(int j=0;j<n.length;j++)
					n[j] = new Neuer(hid_n[i-1],act_fuction);

			hidden_Neuer[i] = n;
		}// end for

	}

	//损失函数
	private double Error(double[] o,double[] t){
		double r = 0;
		for(int i=0;i<o.length;i++){
			r += Loss_function.f(o[i],t[i]);
		}
		return r / t.length;
	}
	
	//交叉熵损失函数
	private double Error_log(double[] o,double[] t){
		double r = 0;
		for(int i=0;i<o.length;i++){
			double r0 = -( t[i]*Math.log10(o[i]) + (1-t[i])*Math.log10(1-o[i]));
			r += r0 * r0;
		}
		return r / (2.0*t.length);
	}

	//获取二维数组中的最大值
	private double getListMax(double[][] data){
		double[] _data = new double[data.length];
		for(int i=0;i<data.length;i++){
			DoubleSummaryStatistics stat = Arrays.stream(data[i]).summaryStatistics();
			// double min = stat.getMin();
			double max = stat.getMax();
			_data[i] = max;
		}

		DoubleSummaryStatistics stat2 = Arrays.stream(_data).summaryStatistics();
		return stat2.getMax();
	}

	//数组连接
	private static double[] merge(double[] a1, double[] a2) {
		double[] a3 = new double[a1.length + a2.length];
		System.arraycopy(a1, 0, a3, 0, a1.length);
		System.arraycopy(a2, 0, a3, a1.length, a2.length);
		return a3;
	}

	//sava
	private String SInt(String name,int vlue){
	    return name+":"+vlue;
	}
	private int getSInt(String infor){
	    return Integer.parseInt( infor.substring(infor.indexOf(":")+1));
	}
	private String Sdouble(String name,double vlue){
	    return name+":"+vlue;
	}
	private double getSdouble(String infor){
	    return Double.parseDouble(infor.substring(infor.indexOf(":")+1));
	}
	private String sIntArrays(String name,int[] data){
		return name + ":length:" + data.length + " " + Arrays.toString(data);
	}
	private String sDoubleArrays(String name,double[] data){
		return name + ":length:" + data.length + " " + Arrays.toString(data);
	}
	
	private int[] getsIntArrays(String s){
		int length = Integer.parseInt( s.substring(s.lastIndexOf(":")+1,s.indexOf("[")-1));
		int[] sd=new int[length];
		int n=0;
		char d = ',';
		for(int i=0;i<s.length();i++){
			if(s.charAt(i)=='[' || s.charAt(i)==','){
				i++;
				String data="";
				while(s.charAt(i)!=d && s.charAt(i)!=']'){
					data += s.charAt(i);
					i++;
				}
				sd[n] =(int) Double.parseDouble(data);
				n++;
				i--;
			}
		}
		return sd;
	}
	
	private double[] getsDoubleArrays(String s){
		int length = Integer.parseInt( s.substring(s.lastIndexOf(":")+1,s.indexOf("[")-1));
		double[] sd=new double[length];
		int n=0;
		char d = ',';
		for(int i=0;i<s.length();i++){
			if(s.charAt(i)=='[' || s.charAt(i)==','){
			  i++;
			  String data="";
			  while(s.charAt(i)!=d && s.charAt(i)!=']'){
				  data += s.charAt(i);
				  i++;
			  }
			  //System.out.println(data);
			  sd[n] = Double.parseDouble(data);
			  n++;
			  i--;
			}
		}
		return sd;
	}

	//向文件追加一行
	private void save_wirter(String s,String path) {
		File f=new File(path);
		if(f.isFile()){
			FileWriter fw = null;
			try {
				fw = new FileWriter(f, true);
		    }catch (IOException e) {
				e.printStackTrace();
		    }
		    PrintWriter pw = new PrintWriter(fw);
		    pw.println(s);
		    pw.flush();
		    try {
				fw.flush();
				pw.close();
				fw.close();
		    } catch (IOException e) {
		     	e.printStackTrace();
			}
		}}
	private void save_wirter(String[] s,String path) {
		File f=new File(path);
		if(f.isFile()){
			FileWriter fw = null;
			try {
				fw = new FileWriter(f, true);
			}catch (IOException e) {
				e.printStackTrace();
			}
			PrintWriter pw = new PrintWriter(fw);
			for(String data:s) pw.println(data);
			pw.flush();
			try {
				fw.flush();
				pw.close();
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}}

	//从文件加载神经网络

	private void readFile(String path){
		File file = new File(path);
		if(file.isFile())
			try {
				FileReader fileReader = null;
				fileReader = new FileReader(file);
				BufferedReader in = new BufferedReader(fileReader);
				String line = in.readLine();
				EXPLAIN = line.substring(8);
				line = in.readLine();
				input_n = getSInt(line);
				line = in.readLine();
				output_n = getSInt(line);
				line = in.readLine();
				nl = getSdouble(line);
				line = in.readLine();
				hid_n = getsIntArrays(line);
				line = in.readLine();
				dError = getSdouble(line);
				line = in.readLine();
				_in_max = getSdouble(line);
				line = in.readLine();
				_out_max = getSdouble(line);
				line = in.readLine();
				act_fuctiom_ID = getSInt(line);
				line = in.readLine();
				Loss_function = getFuctionById(getSInt(line));


				outNeuer_out = new double[output_n];
				hiddenNeuer_out = new double[hid_n.length][];
				for(int i=0;i<hid_n.length;i++)
					hiddenNeuer_out[i] = new double[hid_n[i]];



				output_Neuer = new Neuer[output_n];
				for(int i=0;i<output_n;i++) {
					line = in.readLine();
					Fuction actf = getFuctionById(getSInt(line));
					line = in.readLine();
					double d = getSdouble(line);
					double[] w = new double[hid_n[hid_n.length-1]];
					for(int j=0;j<w.length;j++){
						line = in.readLine();
						w[j] = getSdouble(line);
					}

					Neuer neuer = new Neuer(w.length);
					neuer.w = w;
					neuer.d = d;
					neuer.ACT_function = actf;
					output_Neuer[i] = neuer;
				}

				hidden_Neuer = new Neuer[hid_n.length][];
				for (int i=0;i<hid_n.length;i++)
					hidden_Neuer[i] = new Neuer[hid_n[i]];
				//第0层隐藏层
				for(int i=0;i<hidden_Neuer[0].length;i++){
					line = in.readLine();
					Fuction actf = getFuctionById(getSInt(line));
					line = in.readLine();
					double d = getSdouble(line);
					double[] w = new double[input_n];
					for(int j=0;j<w.length;j++){
						line = in.readLine();
						w[j] = getSdouble(line);
					}
					Neuer neuer = new Neuer(w.length);
					neuer.w = w;
					neuer.d = d;
					neuer.ACT_function = actf;
					hidden_Neuer[0][i] = neuer;
				}

				for(int n=1;n<hidden_Neuer.length;n++)//第n层
					for(int i=0;i<hidden_Neuer[n].length;i++){
						line = in.readLine();
						Fuction actf = getFuctionById(getSInt(line));
						line = in.readLine();
						double d = getSdouble(line);
						double[] w = new double[hid_n[n-1]];
						for(int j=0;j<w.length;j++){
							line = in.readLine();
							w[j] = getSdouble(line);
						}
						Neuer neuer = new Neuer(w.length);
						neuer.w = w;
						neuer.d = d;
						neuer.ACT_function = actf;
						hidden_Neuer[n][i] = neuer;
					}

		        in.close();
		        fileReader.close();
	       }catch(Exception e){ System.out.println(e); }
    }


    /* old
	private void readFile(String path){
		File file = new File(path);
		if(file.isFile())
			try {
				FileReader fileReader = null;
				fileReader = new FileReader(file);
				BufferedReader in = new BufferedReader(fileReader);
				String line = in.readLine();
				EXPLAIN = line.substring(8);
				line = in.readLine();
				input_n = getSInt(line);
				line = in.readLine();
				output_n = getSInt(line);
				line = in.readLine();
				nl = getSdouble(line);
				line = in.readLine();
				hid_n = getsIntArrays(line);
				line = in.readLine();
				dError = getSdouble(line);
				line = in.readLine();
				_in_max = getSdouble(line);
				line = in.readLine();
				_out_max = getSdouble(line);

				inNeuer_out = new double[input_n];
				outNeuer_out = new double[output_n];
				hiddenNeuer_out = new double[hid_n.length][];
				for(int i=0;i<hid_n.length;i++)
					hiddenNeuer_out[i] = new double[hid_n[i]];

				input_Neuer = new Neuer[input_n];
				for(int i=0;i<input_n;i++){
					Neuer neuer = new Neuer();
					line = in.readLine();
					neuer.d = getSdouble(line);
					line = in.readLine();
					neuer.w[0] = getSdouble(line);
					neuer.ACT_function = new Sigmoid();
					input_Neuer[i] = neuer;
				}

				output_Neuer = new Neuer[output_n];
				for(int i=0;i<output_n;i++) {
					line = in.readLine();
					double d = getSdouble(line);
					double[] w = new double[hid_n[hid_n.length-1]];
					for(int j=0;j<w.length;j++){
						line = in.readLine();
						w[j] = getSdouble(line);
					}

					Neuer neuer = new Neuer(w.length);
					neuer.w = w;
					neuer.d = d;
					neuer.ACT_function = new Sigmoid();
					output_Neuer[i] = neuer;
				}

				hidden_Neuer = new Neuer[hid_n.length][];
				for (int i=0;i<hid_n.length;i++)
					hidden_Neuer[i] = new Neuer[hid_n[i]];
				//第0层隐藏层
				for(int i=0;i<hidden_Neuer[0].length;i++){
					line = in.readLine();
					double d = getSdouble(line);
					double[] w = new double[input_n];
					for(int j=0;j<w.length;j++){
						line = in.readLine();
						w[j] = getSdouble(line);
					}
					Neuer neuer = new Neuer(w.length);
					neuer.w = w;
					neuer.d = d;
					neuer.ACT_function = new Sigmoid();
					hidden_Neuer[0][i] = neuer;
				}

				for(int n=1;n<hidden_Neuer.length;n++)//第n层
					for(int i=0;i<hidden_Neuer[n].length;i++){
						line = in.readLine();
						double d = getSdouble(line);
						double[] w = new double[hid_n[n-1]];
						for(int j=0;j<w.length;j++){
							line = in.readLine();
							w[j] = getSdouble(line);
						}
						Neuer neuer = new Neuer(w.length);
						neuer.w = w;
						neuer.d = d;
						neuer.ACT_function = new Sigmoid();
						hidden_Neuer[n][i] = neuer;
					}

				in.close();
				fileReader.close();
			}catch(Exception e){ System.out.println(e); }
	}

     */

	private Fuction getFuctionById(int id){
		Fuction r;
		switch (id){
			case 1: r = new Sigmoid(); break;
			case 2: r = new Tanh();    break;
			case 3: r = new Relu();    break;
			case 4: r = new LRelu();   break;

			case 10:r = new MSELoss(); break;
			case 11:r = new CELoss();  break;
			case 12:r = new CESLoss(); break;
			default:r = new Fuction(); break;
		}
		return r;
	}

	//获取神经网络的参数数量
	public int getWandD_number(){
		//输入层
		int r = 0;//input_Neuer.length + input_Neuer.length * input_Neuer[0].w.length;

		//隐藏
		for (int i=0; i<hidden_Neuer.length;i++){
			r += hidden_Neuer[i].length + hidden_Neuer[i].length * hidden_Neuer[i][0].w.length;
		}
		r += output_Neuer.length + output_Neuer[0].w.length;
		return r;
	}
}




