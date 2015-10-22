import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.TreeSet;
/*
 * N: Size of training vocab
 * Features (0): Class
 * Features (1..N): Bool checks on word in sentence
 * Features (N+1:N+len(C)): Collocation vals (Value is index of word in sorted vocab + 1)
 */

public class sctrain {
	private static final String STOP_WORDS_PATH = "stopwd.txt";
	private static TreeMap<String, Boolean> _stopWordsMap;
	private static int _leftOffset = -3;
	private static int _rightOffset = 3;
	private static final String START_TOKEN = "<START>";
	private static final String END_TOKEN = "<END>";
	private static final String NULL_TOKEN = "<NULL>";
	private static TreeSet<String> _vocabSet = new TreeSet<String>();
	private static HashMap<String, Integer> _vocabLookup = new HashMap<String, Integer>();
	private static final int NUM_ITERS = 2000;
	private static final double LEARNING_RATE = 0.1f;

	private static void populateVocab(ArrayList<String[]> sentences) {
		for (String[] sentence : sentences) {
			for (String word : sentence) {
				if (!isSpecialToken(word))
				_vocabSet.add(word);
			}
		}
		
		int i = 1;
		for (String word : _vocabSet) {
			_vocabLookup.put(word, i++);
		}
		
		for(String stopWord : _stopWordsMap.keySet()) {
			_vocabLookup.put(stopWord, i++);
		}
		
		_vocabLookup.put(START_TOKEN, i++);
		_vocabLookup.put(END_TOKEN, i++);
		_vocabLookup.put(NULL_TOKEN, i++);
		
	}

	private static boolean isSpecialToken(String word) {
		return word.equals(START_TOKEN) || word.equals(END_TOKEN) || word.equals(NULL_TOKEN);
	}

	private static ArrayList<String[]> getTrainSentencesWithNoStops(ArrayList<String[]> trainSentences) {
		ArrayList<String[]> trainSentencesNoStops = new ArrayList<String[]>();
		
		for (String[] sentence : trainSentences) {
			ArrayList<String> sentenceNoStops = new ArrayList<String>();
			for (String word : sentence) {
				if (!_stopWordsMap.containsKey(word)) {
					sentenceNoStops.add(word);
				}
			}

			trainSentencesNoStops.add(sentenceNoStops.toArray(new String[sentenceNoStops.size()]));
		}
		
		return trainSentencesNoStops;
	}
	
	private static void populateStopWordMap() {

		_stopWordsMap = new TreeMap<String, Boolean>();
		try (BufferedReader br = new BufferedReader(new FileReader(STOP_WORDS_PATH)))
		{
			String sCurrentLine;

			while ((sCurrentLine = br.readLine()) != null) {
				_stopWordsMap.put(sCurrentLine.trim(), true);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static ArrayList<String[]> getSanitizedTrainingExamples(ArrayList<String> trainSentences) {
		ArrayList<String[]> sanitizedExamples = new ArrayList<String[]>();
		
		for (String sentence : trainSentences) {
			String[] words = sentence.substring(sentence.indexOf('\t')).replaceAll("[^a-zA-Z0-9 ]", "").toLowerCase().trim().split("\\s+");
			String[] allWords = new String[(words.length + 2)];
			System.arraycopy(words, 0, allWords, 1, words.length);
			allWords[0] = START_TOKEN;
			allWords[words.length + 1] = END_TOKEN;
			sanitizedExamples.add(allWords);
		}
		
		return sanitizedExamples;
	}
	
	public static ArrayList<String> readTrainFile(String trainPath) {
		ArrayList<String> sentences = new ArrayList<String>();

		try (BufferedReader br = new BufferedReader(new FileReader(trainPath)))
		{

			String sCurrentLine;

			while ((sCurrentLine = br.readLine()) != null) {
				sentences.add(sCurrentLine);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return sentences;
	}
	
	public static void main(String[] args) {
		String classA = args[0];
		String classB = args[1];
		String trainPath = args[2];
		String modelPath = args[3];
		
		populateStopWordMap();
		ArrayList<String[]> trainSentences = getSanitizedTrainingExamples(readTrainFile(trainPath));
		ArrayList<String[]> trainSentencesNoStops = getTrainSentencesWithNoStops(trainSentences);
		populateVocab(trainSentencesNoStops);
		
		ArrayList<double[]> featureArrays = getFeatureArrays(classA, classB, trainSentences, trainSentencesNoStops);
		
		double[] initWeights = new double[featureArrays.get(0).length];
		
		train(NUM_ITERS, initWeights, featureArrays);
		
		writeOutput(modelPath, initWeights, classA, classB);
		
	}

	private static void writeOutput(String modelPath, double[] initWeights, String classA, String classB) {

		try (PrintWriter writer = new PrintWriter(modelPath, "UTF-8"))
		{
			writer.println(_leftOffset + " " + _rightOffset);
			String sstr = "";
			for (String w : _stopWordsMap.keySet()) {
				sstr += w + " ";
			}
			writer.println(sstr.trim());
			
			String vstr = "";
			for (String w : _vocabSet) {
				vstr += w + " ";
			}
			writer.println(vstr.trim());
			
			String vlstr = "";
			for (String w : _vocabLookup.keySet()) {
				vlstr += w + " " + _vocabLookup.get(w) + " ";
			}
			
			writer.println(vlstr.trim());
			
			String wstr = "";
			for (double w : initWeights) {
				wstr += w + " ";
			}
			writer.println(wstr.trim());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void train(int numIters, double[] initWeights, ArrayList<double[]> featureArrays) {
		
		int i = 0;
		while (i++ < numIters) {
			//System.out.println("Running Iter " + i);
			for (double[] featureArray : featureArrays) {
				updateWeights(initWeights, featureArray);
			}
		}
	}

	private static void updateWeights(double[] initWeights, double[] featureArray) {
		assert initWeights.length == featureArray.length;
		
		initWeights[0] = initWeights[0] + LEARNING_RATE * (featureArray[0] - logistic(initWeights[0]));
		
		double dotProduct = getDotProduct(initWeights, featureArray, 1);
		
		for (int i = 1; i < featureArray.length; i++) {
			initWeights[i] = initWeights[i] + (LEARNING_RATE * featureArray[i]) * (featureArray[0] - logistic(dotProduct)); 
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

	private static ArrayList<double[]> getFeatureArrays(String classA, String classB, ArrayList<String[]> trainSentences,
			ArrayList<String[]> trainSentencesNoStops) {
		ArrayList<double[]> featureArrays = new ArrayList<double[]>();
		
		int collocatedItems = getNumCollocatedItems();
		assert collocatedItems >= 0;
		
		for (int i = 0; i < trainSentences.size(); i++) {
			String[] trainSentence = trainSentences.get(i);
			String[] trainSentenceNoStop = trainSentencesNoStops.get(i);
			
			double[] featureArray = new double[(1 + _vocabSet.size()) + (_vocabLookup.keySet().size() * collocatedItems) + 1];
			
			featureArray[0] = getExClass(trainSentence, classA, classB);
			assert featureArray[0] != -1;
			
			for (String word : trainSentenceNoStop) {
				if (isSpecialToken(word) || word.equals(classA) || word.equals(classB)) {
					continue;
				}
				featureArray[_vocabLookup.get(word)] = 1;
			}
			
			int classWordLocation = getClassWordLocation(trainSentence, classA, classB);
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
				if (sumIndex < 0 || sumIndex >= trainSentence.length) {
					featureArray[fIndexBase + _vocabLookup.get(NULL_TOKEN)] = 1.0f;
				} else {
					featureArray[fIndexBase + _vocabLookup.get(trainSentence[sumIndex])] = 1.0f;
				}
				
				iter++;
				colIndex++;
			}
			
			featureArrays.add(featureArray);
		}
		
		return featureArrays;
	}

	private static int getClassWordLocation(String[] trainSentence, String c1, String c2) {
		for (int i = 0; i < trainSentence.length; i++) {
			String s = trainSentence[i];
			
			if (s.equalsIgnoreCase(c1) || s.equalsIgnoreCase(c2)) {
				return i;
			}
		}
		
		// Error Finding Class
		System.out.println("Class Not Found For " + trainSentence);
		return -1;
	}

	private static int getExClass(String[] words, String c1, String c2) {
		for (String s : words) {
			if (s.equalsIgnoreCase(c1)) {
				return 0;
			} else if (s.equalsIgnoreCase(c2)) {
				return 1;
			}
		}
		
		// Error Finding Class
		System.out.println("Class Not Found For " + words);
		return -1;
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
}
