The purpose of this directory is to define a test set to validate the sufficient and the necessary features determined by the algorithm.
The goal is to check if similar sentences produce the same sufficient and necessary features.

This directory contains two folders
- "qa" folder : for question answering sentences
- "sa" folder: for sentiment analysis sentences

each directory contains a set of files. Each file contains a set of similar sentences and has the following structure :
- The first line contains the id of the sentence in the dataset
- The second line contains three fields separated by ":", the first field is the original sentence, the second field is the sufficient phrase
  detected by the explanation algorithm and the third field represent the list of necessary features
- The following lines consists of the variations of the original sentence that preserve the sufficient phrase and the necessary phrases.