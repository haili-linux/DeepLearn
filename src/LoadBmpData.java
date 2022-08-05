import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class LoadBmpData {

    /**
     * 创建随机数据集
     * @param len 数据集长度
     * @param vector 维度
     * @return 数据集
     */
    public static double[][] createDataSet(int len, int vector){
        vector++;
        double[][] train = new double[len][vector];

        int r = train.length/2;
        for(int i=0; i < train.length; i++){

            if(i < r){
                train[i][vector-1] = 0;
                for (int j = 0; j < vector-1; j++) train[i][j] = Math.random()*5; //0~10
            } else {
                train[i][vector-1] = 1;
                for (int j = 0; j < vector-1; j++) train[i][j] = 5 + Math.random()*5; //11~20
            }
        }
        return train;
    }

    /**
     * bmp图片像素转数组
     * @param bmpFilePath bmp图片路径
     * @param label 该图片的标签
     * @return 。
     * @throws Exception null
     */
    public static double[] bmpToRgbList(String bmpFilePath, int label) throws Exception{
        File bmpFile = new File(bmpFilePath);
        BufferedImage image = ImageIO.read(bmpFile);
        int height = image.getHeight();
        int width = image.getWidth();

        //rgb数组，多一位放标签label
        double[] rgb = new double[ height * width + 1];
        int index = 0;

        int pixel;
        for(int i = 0; i < width; i++)
            for(int j = 0; j < height; j++){
                pixel = image.getRGB(i,j);
                rgb[index] = (pixel &  0xff);
                index++;
            }
        rgb[index] = label;
        return rgb;
    }
    public static double[] bmpToRgbList(String bmpFilePath) throws Exception{
        File bmpFile = new File(bmpFilePath);
        BufferedImage image = ImageIO.read(bmpFile);
        int height = image.getHeight();
        int width = image.getWidth();

        //rgb数组，多一位放标签label
        double[] rgb = new double[ height * width];
        int index = 0;

        int pixel;
        for(int i = 0; i < width; i++)
            for(int j = 0; j < height; j++){
                pixel = image.getRGB(i,j);
                rgb[index] = (pixel &  0xff);
                index++;
            }
        return rgb;
    }
    public static double[][] load_Data(String path) throws Exception{

        File file = new File(path);

        if (!file.exists() || file.isFile()) return null;

        File[] file2 = file.listFiles();

        if (file2 == null) return null;


        int len = 0;

        File[][] label_file =new File[file2.length][];
        for(int i = 0; i < label_file.length; i++){
            label_file[i] = file2[i].listFiles();
            len += label_file[i].length;
        }

        double[][] data = new double[len][];

        int index = 0;
        for(int i = 0; i < label_file.length; i++ ){
            for(int j = 0; j < label_file[i].length; j++ ){
                data[index] = bmpToRgbList( label_file[i][j].toString(),i);
                index++;
            }
        }


//        for(int i = 0; i < label_0.length; i++ ){
//            data[i] = bmpToRgbList( label_0[i].toString(),0);
//            index = i;
//        }
//
//        index++;
//
//        for(int i=0; i <label_1.length; i++){
//            data[ i+index ] = bmpToRgbList( label_1[i].toString(),1);
//        }
        return data;
    }
}
