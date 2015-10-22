package cs4248;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.TreeSet;

public class sctest {
	private static final String STOP_WORDS_PATH = "./files/stopwd.txt";
	private static TreeMap<String, Boolean> _stopWordsMap = new TreeMap<String, Boolean>();
	private static final String START_TOKEN = "<START>";
	private static final String END_TOKEN = "<END>";
	private static final String NULL_TOKEN = "<NULL>";
	private static final String PLACEHOLDER = "placeholderfortestcase";
	private static int _leftOffset;
	private static int _rightOffset;
	private static TreeSet<String> _vocabSet = new TreeSet<String>();
	private static HashMap<String, Integer> _vocabLookup = new HashMap<String, Integer>();
	private static double _weights[];
	private static TreeMap<String, String[]> _testCases = new TreeMap<String, String[]>();
	private static TreeMap<String, String[]> _testCasesNoStops = new TreeMap<String, String[]>();
	
	public static void main(String args[]) {
		String classA = args[0];
		String classB = args[1];
		String testFilePath = args[2];
		String modelPath = args[3];
		String outputPath = args[4];
		
		loadModel(modelPath);
		loadTests(testFilePath);
		
		ArrayList<String> output = new ArrayList<String>();
		
		for (String tId : _testCases.keySet()) {
			double[] featureArray = getFeatureArray(tId);
			
			double p = logistic(getDotProduct(_weights, featureArray, 1));
			output.add(tId + "\t" + getClassFromProb(p, classA, classB));
		}
		
		writeOutput(outputPath, output);
	}
	
	private static void writeOutput(String outputPath, ArrayList<String> output) {
		try (PrintWriter writer = new PrintWriter(outputPath, "UTF-8"))
		{
			for(String s : output) {
				writer.println(s);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static String getClassFromProb(double p, String c1, String c2) {
		if (p >= 0.5) {
			return c2;
		}
		return c1;
	}
	
	private static boolean isSpecialToken(String word) {
		return word.equals(START_TOKEN) || word.equals(END_TOKEN) || word.equals(NULL_TOKEN);
	}
	
	private static double[] getFeatureArray(String tId) {
		
		int collocatedItems = getNumCollocatedItems();
		assert collocatedItems >= 0;
		String[] sentence = _testCases.get(tId);
		String[] sentenceNoStop = _testCasesNoStops.get(tId);
			
		double[] featureArray = new double[(1 + _vocabSet.size()) + (_vocabLookup.keySet().size() * collocatedItems) + 1];
			
		for (String word : sentenceNoStop) {
			if (isSpecialToken(word) || word.equals(PLACEHOLDER)) {
					continue;
			}
			if (!_vocabLookup.containsKey(word)) {
				//System.out.println("OOV: " + word);
				continue;
			}
			featureArray[_vocabLookup.get(word)] = 1;
		}
			
		int classWordLocation = getPlaceholderLocation(sentence);
		assert classWordLocation != -1;
			
		int colIndex = _leftOffset;
		int iter = 0;
			
		while (iter < getNumCollocatedItems()) {
				
			int sumIndex = classWordLocation + colIndex;
				
			if (sumIndex == classWordLocation) {
				iter++;
				colIndex++;
				continue;
			}
				
			int fIndexBase = 1 + _vocabSet.size() + (iter * _vocabLookup.size());
			if (sumIndex < 0 || sumIndex >= sentence.length) {
				featureArray[fIndexBase + _vocabLookup.get(NULL_TOKEN)] = 1.0f;
			} else if (_vocabLookup.containsKey(sentence[sumIndex])){
				featureArray[fIndexBase + _vocabLookup.get(sentence[sumIndex])] = 1.0f;
			}
				
			iter++;
			colIndex++;
		}
		
		return featureArray;
	}
	
	private static int getPlaceholderLocation(String[] sentence) {
		for (int i = 0; i < sentence.length; i++) {
			String s = sentence[i];
			
			if (s.equalsIgnoreCase(PLACEHOLDER)) {
				return i;
			}
		}
		
		// Error Finding Class
		System.out.println("Placeholder Not Found For " + sentence);
		return -1;
	}

	private static void loadTests(String testFilePath) {
		try (BufferedReader br = new BufferedReader(new FileReader(testFilePath))) {
			String sentence;
			
			while ((sentence = br.readLine()) != null) {
				sentence = sentence.replaceAll(">> <<", PLACEHOLDER);
				String[] words = sentence.substring(sentence.indexOf('\t')).replaceAll("[^a-zA-Z0-9 ]", "").toLowerCase().trim().split("\\s+");
				String[] allWords = new String[(words.length + 2)];
				System.arraycopy(words, 0, allWords, 1, words.length);
				allWords[0] = START_TOKEN;
				allWords[words.length + 1] = END_TOKEN;
				String tId = sentence.substring(0, sentence.indexOf('\t'));
				_testCases.put(tId, allWords);
				_testCasesNoStops.put(tId, getSentenceWithNoStops(allWords));
				
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	private static String[] getSentenceWithNoStops(String[] sentence) {
		ArrayList<String> sentenceNoStops = new ArrayList<String>();
		for (String word : sentence) {
			if (!_stopWordsMap.containsKey(word)) {
				sentenceNoStops.add(word);
			}
		}

		return sentenceNoStops.toArray(new String[sentenceNoStops.size()]);
	}
	
	private static int getNumCollocatedItems() {
		if (_leftOffset == 0 && _rightOffset == 0) {
			return 0;
		} else if ((_leftOffset * _rightOffset) > 0) {
			return _rightOffset - _leftOffset + 1;
		} else {
			return _rightOffset - _leftOffset;
		}
	}
	
	private static double getDotProduct(double[] initWeights, double[] featureArray, int start) {
		double dotProduct = 0;
		for (int i = start; i < initWeights.length; i++) {
			dotProduct += (initWeights[i] * featureArray[i]);
		}
		return dotProduct;
	}

	private static double logistic(double power) {
		return 1.0 / (1 + Math.exp((-1 * power)));
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
