# Finding word similarity using Skip-gram and negative sampling 
* Finding similarity of words is one of the basic problems in the natural language processing field. In this problem, we must define a vector for each word, then calculate the cosine distance between them.
* We used skip-gram and negative sampling to calculate the weights of each word.

## Prerequisites
Install below packages:
* argparse
* Pandas
* Spacy 
* tqdm
* Numpy 
* Sklearn

## Steps of the model
```
Words -> Preprocessing -> Make dictionary -> Create unigram and Negative sampling -> Training -> Calculate similarity
```

## Preprocessing
The document is first split into sentences, the sentences are converted into word tokens. Since the presence of punctuation is not necessary for our task we eliminate them using a predefined
spacy function. The result of the preprocessing steps is a list of tokens from each sentence that are void of punctuations. 

## Make a dictionary with ID 
We create a dictionary where the keys are the words and the values are the number of occurrences of the word. This is defined in the init function and used in various parts of the program 
to see the length of our vocabulary and to loop through all the words of the given corpus. We also have two functions 1)w2id which given the word returns the index of the word 2)id2cnt 
which provides the mapping between the word index and the count of the word. We also remove the words which occur less than the min-count argument. 

## Create unigram and selecting random words. 
To randomly sample the words from the given corpus, we create the unigram table where the keys are the words and the values are the probability of occurrence of each word.
To ensure the words with maximum occurrence do not get sampled the majority of the time, we raise the probability of each word to 3/4 which flattens the probability curve
ensuring more unique words get sampled. With this new probability distribution, we randomly sample a few words for each target and context words. 

## Training
For negative sampling, we use a sigmoid function as the activation function in the neural network. 
For each word we intend to train on we pass the (wordid, context word id, negative sample for each contxt Id) to the train word function.
We use gradient descent for updating the weights in each iteration. Our understanding of the gradients and weights update are obtained from the Algorithm 1 SGNS Word2Vec((Stergiou et al., 2017)
In brief, we use two weight-matrixes(w1 - input weight and w2 - output weight). All our context words we take as positive words(label=1) and all the negative sampled words we take as negative words(label=0).
We now can have a common equation for updating both w1 and w2 (Mentioned in the report).

### Update procedure
forward propogate
1) Find the dot-product of the w1(indexing the word) and w2(indexing the current negative word)
2) Apply the activation function on the output(sigmoid), it's our result to compare with true labels to get prediction error.
backpropagate
3) Derivatie of loss function by w2 is equal to prediction error multiplied by w1 and derivatie of loss function by w1 is equal to prediction error multiplied by w2.
4)Update w2 of positive and negative words using previous output weight and learning rate multiplied by derivative of loss function by w2. 
5)Update only w1 of current target word using previous input weight and learning rate multiplied by derivative of loss function by w1. After looping through all the words, we can update all w1.



## Calculate the similarity

After training, now have the w1 which is the vector representation of the unique words in the given corpus. Therefore we can obtain the vector representation of each word by first finding
index using the w2id function and then using this value to index w1. After performing the same for both we obtain the cosine distance between the two vectors to calculate the similarity.
In case the word is not present in the corpus on which data is trained, 0 is returned as similarity.

## Basic Usage
For Running, type in terminal
```
python skipGramModel.py

```

Please note
1)We have attached a ipynb file to show our implementation and understanding of the basic skipgram model
2)The regex in the test script throws an error and we had a difficult time running the test script 
We use the regex "[-+]?[0-9]*\.?[0-9]*$" and it works for us. The delay of 10mins was due to this, I hope you will consider our situation. 