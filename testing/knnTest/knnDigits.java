import java.time.format.DateTimeFormatter;  
import java.time.LocalDateTime;   
import java.util.*;
import java.lang.Math;
import java.io.*;

public class knnDigits {

	public static ArrayList<float[]> trainingData = new ArrayList();
	public static ArrayList<float[]> testData = new ArrayList();  
	public static float results[];
	public static double trainingConst = Math.floor(42000*.2);
	public static double k = 9;
	public static PrintWriter writer; 

	public static int readData(){

		BufferedReader csvReader;
		String row;	
		try {
			//read file by line
			csvReader = new BufferedReader(new FileReader("./digit-recognizer/train.csv"));
			row = csvReader.readLine();

			//enter data
			double p = trainingConst;
			while ((row = csvReader.readLine()) != null && p >= 0) {
				String[] data = row.split(",");
				float[] floatData = new float[data.length];
				for (int i = 0; i < data.length; i++){
					floatData[i] = Float.parseFloat(data[i]);
				}
				
				trainingData.add(floatData);		
				p--;
			}
			while ((row = csvReader.readLine()) != null) {
                                String[] data = row.split(",");
                                float[] floatData = new float[data.length];
                                for (int i = 0; i < data.length; i++){
                                        floatData[i] = Float.parseFloat(data[i]);
                                }

                                testData.add(floatData);
                        }

			csvReader.close();
		}
                catch (Exception e) {
                        System.out.println(e);
                        return 1;
                }

		
		results = new float[testData.size()];
		return 0;
	}

	//p power to raise
	public static void Minkowski(int p){

		for (int i = 0; i<testData.size(); i++){
			
			//array to hold distances and values
			float dist[][] = new float [trainingData.size()][2];

			for (int j = 0; j<trainingData.size(); j++){
				float sum = (float)0.0;
				
				//distance excluding the result
				for (int r = 1; r<trainingData.get(0).length; r++){
					sum += Math.abs( Math.pow( (testData.get(i)[r] - trainingData.get(j)[r]) , p) );
				}
				
				sum = (float)Math.pow(sum, (1.0/(double)p));
				dist[j][0] = sum;
				dist[j][1] = trainingData.get(j)[0];
			}

			dist = sortDist(dist);
			
			int mode[] = new int[10];
			int maxCount = 0;
			int maxValue = 0;
			for (int j = 0; j<k; j++){
				mode[(int)dist[j][1]]++;
				if (mode[(int)dist[j][1]] > maxValue){
					maxValue = mode[(int)dist[j][1]];
					maxCount = (int)dist[j][1];
				}
			}
			
			results[i] = maxCount;
		}
	}
	
	//sort by distance
	public static float[][] sortDist(float[][] d){
		//minimum values at the top
		for (int i = 0; i < d.length; i++){
			for (int j = i+1; j < d.length; j++){
			
				if ( d[j][0] < d[i][0] ) {
					float[] temp = d[j];
					for (int y = j; y > i; y--){
						d[y] = d[y-1]; 		
					}	
					d[i] = temp;
				}
			
			}
		}	
		return d;
	}

	
	public static void errorReport(){

		int confusionMatrix[][] = new int [10][10];
		for (int i = 0; i<testData.size(); i++){
			confusionMatrix[(int)results[i]][(int)testData.get(i)[0]]++;
		}
	
		for (int i = 0; i<confusionMatrix.length; i++){
			for (int j = 0; j<confusionMatrix.length; j++){
				System.out.print(confusionMatrix[i][j]);
				writer.print(confusionMatrix[i][j]);	
				if (j != 9){
					System.out.print(',');
					writer.print(',');
				}
			}
			System.out.println();
			writer.println();
		}
		System.out.println();
		

		//for the number 1	
		System.out.println("True Positives: " + confusionMatrix[1][1] );

		int tn = 0;
		for (int i = 0; i<confusionMatrix.length; i++){		
			tn += confusionMatrix[i][i];
		}
		System.out.println("True Negatives: " + tn);

		int fp = 0;
		for (int i = 0; i<confusionMatrix.length; i++){
                	if (i != 1){
		       		fp += confusionMatrix[1][i];
			}
                }	
		System.out.println("False Positives: " + fp);
	
		int fn = 0;
		for (int i = 0; i<confusionMatrix.length; i++){
                        if (i != 1){
                                fp += confusionMatrix[i][1];
                        }
                }	
		System.out.println("False Negatives: " + fn);
	}

	public static void main(String[] args){
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");  
   		LocalDateTime start = LocalDateTime.now();  

		if (readData() == 1){
			System.out.println("Error: file not found. Exiting...");
			return;
		}

		try {
			writer = new PrintWriter("k-digit-variance.csv", "UTF-8");
		}
		catch (Exception e){
			System.out.println(e);
			return;
		}
		//for (k = 1; k<=15; k+=2){
			System.out.println("K EQUALS : " + k);
			writer.println(k);
			Minkowski(1);
			errorReport();
		//}
		writer.close();
		LocalDateTime stop = LocalDateTime.now();
		System.out.println();
		System.out.println(dtf.format(start) + " // " + dtf.format(stop));
        }	
}
