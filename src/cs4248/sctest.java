package cs4248;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.TreeSet;

public class sctest {
	private static final String STOP_WORDS_PATH = "./files/stopwd.txt";
	private static TreeMap<String, Boolean> _stopWordsMap = new TreeMap<String, Boolean>();
	private static final String START_TOKEN = "<START>";
	private static final String END_TOKEN = "<END>";
	private static final String NULL_TOKEN = "<NULL>";
	private static int _leftOffset;
	private static int _rightOffset;
	private static TreeSet<String> _vocabSet = new TreeSet<String>();
	private static HashMap<String, Integer> _vocabLookup = new HashMap<String, Integer>();
	private static double _weights[];
	
	public static void main(String args[]) {
		String classA = args[0];
		String classB = args[1];
		String testFilePath = args[2];
		String modelPath = args[3];
		String outputPath = args[4];
		
		loadModel(modelPath);
		
	}

	private static void loadModel(String modelPath) {
		int lId = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(modelPath)))
		{
			String sCurrentLine;
			
			while ((sCurrentLine = br.readLine()) != null) {
				if (lId == 0) {
					String[] offsets = sCurrentLine.split("\\s+");
					_leftOffset = Integer.parseInt(offsets[0]);
					_rightOffset = Integer.parseInt(offsets[1]);
					
					lId++;
				} else if (lId == 1) {
					String[] vset = sCurrentLine.split("\\s+");
					for (String w : vset) {
						_stopWordsMap.put(w, true);
					}
					lId++;
				} else if (lId == 2) {
					String[] wset = sCurrentLine.split("\\s+");
					for (String w : wset) {
						_vocabSet.add(w);
					}
					lId++;
				} else if (lId == 3) {
					String[] wset = sCurrentLine.split("\\s+");
					for (int i = 0; i < wset.length; i=i+2) {
						_vocabLookup.put(wset[i], Integer.parseInt(wset[i+1]));
					}
					lId++;
				} else if (lId == 4) {
					String[] wset = sCurrentLine.split("\\s+");
					_weights = new double[wset.length];
					for (int i = 0; i < wset.length; i++) {
						_weights[i] = Double.parseDouble(wset[i]);
					}
					lId++;
				}
				
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
