package cs4248;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
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
	private static final String _stopWordsPath = "./files/stopwd.txt";
	private static TreeMap<String, Boolean> _stopWordsMap;
	private static int _leftOffset = -2;
	private static int _rightOffset = 2;
	private static final String _startToken = "<START>";
	private static final String _endToken = "<END>";
	private static final String _nullToken = "<NULL>";
	private static TreeSet<String> _vocabSet = new TreeSet<String>();
	private static HashMap<String, Integer> _vocabLookup = new HashMap<String, Integer>();

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
		
		_vocabLookup.put(_startToken, i++);
		_vocabLookup.put(_endToken, i++);
		_vocabLookup.put(_nullToken, i++);
		
	}

	private static boolean isSpecialToken(String word) {
		return word.equals(_startToken) || word.equals(_endToken) || word.equals(_nullToken);
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
		try (BufferedReader br = new BufferedReader(new FileReader(_stopWordsPath)))
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
			String[] words = sentence.replaceAll("[^a-zA-Z ]", "").toLowerCase().trim().split("\\s+");
			String[] allWords = new String[(words.length + 2)];
			System.arraycopy(words, 0, allWords, 1, words.length);
			allWords[0] = _startToken;
			allWords[words.length + 1] = _endToken;
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
		
		ArrayList<float[]> featureArrays = getFeatureArrays(classA, classB, trainSentences, trainSentencesNoStops);
		
	}

	private static ArrayList<float[]> getFeatureArrays(String classA, String classB, ArrayList<String[]> trainSentences,
			ArrayList<String[]> trainSentencesNoStops) {
		ArrayList<float[]> featureArrays = new ArrayList<float[]>();
		
		int collocatedItems = getNumCollocatedItems();
		assert collocatedItems >= 0;
		
		for (int i = 0; i < trainSentences.size(); i++) {
			String[] trainSentence = trainSentences.get(i);
			String[] trainSentenceNoStop = trainSentencesNoStops.get(i);
			
			float[] featureArray = new float[_vocabSet.size() + 1 + collocatedItems]; // + class + coll
			
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
			int fIndex = _vocabSet.size() + 1;
			
			while (fIndex < featureArray.length) {
				int sumIndex = classWordLocation + colIndex;
				
				if (sumIndex == classWordLocation) {
					colIndex++;
					continue;
				}
				
				if (sumIndex < 0 || sumIndex >= trainSentence.length) {
					featureArray[fIndex++] = (float) (_vocabLookup.get(_nullToken) * 1.0 / _vocabSet.size());
				} else {
					featureArray[fIndex++] = (float) (_vocabLookup.get(trainSentence[sumIndex]) * 1.0 / _vocabSet.size());
				}

				colIndex++;
			}
			
			featureArrays.add(featureArray);
		}
		
		for (float[] fArray : featureArrays) {
			for (float i : fArray) {
				System.out.print(i + " ");
			}
			System.out.println();
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
		if (_leftOffset == 0 || _rightOffset == 0) {
			return 0;
		} else if ((_leftOffset * _rightOffset) > 0) {
			return _rightOffset - _leftOffset + 1;
		} else {
			return _rightOffset - _leftOffset;
		}
	}

	

}
