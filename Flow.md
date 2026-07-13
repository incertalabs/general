work our way up from first principles 

# Act I: How text becomes geometry and machine readable

MNIST - digits
Text - Tokenizer, same idea. Latent space
- Tiktokenizer example - why alot of the problem from large language models come from their tokenizer. 
- What the LLM actually see's and how it is a learned statistical representation of the last word. 
- How this works for images 
--
Start where a statisitican stars, what the unit of observation before we can even ask what the model is
- For CPSC we work with documents or reports = variable-length string of text symbols. 
Well, how do we turn these symbols of strings into somethign a computer cna read? 
Any modeling task we want to run is always going to need a fixed-length vector $x\in\mathbb{R}^p$. 
- Read: x is a element of R p, 
- Vector X is a list of p real numbers
- Our vector size, that letter p, is the size of words/text/tokens in our vocab. 
Now we have to make a important decisions. 
I'm going to introduce 3 things. 
1) How clssical ML like IF-IDF, which we covered in class its just XXXXXXXXXXXXXXXXX, break up there words. 
2) How modern LLM's do it - bypte-pair toekns 
	1) show tiktoneize 
	2) Talk about how alot of the problems before with LLM's come with tokensization such as: give a brief tease about how this becomes very important later with LLMs. 
	3) THe less words we can use to 
	4) Show how koneizie and helloe are 1 vs 6 tokens. How you'd need much more compute 
	5) and why the internet is also all english. But its getting better. It all comes down to tkensization and alot of the problems we saw with LLMs were l ike this 
	6) Often old LLMs would thing that 3.11 was larger than 3.9 and it was always becuase of how the tokens were borken up. 
	7) 

3) How we represents images in vector space 
	1) minist example, feature space of 



# Act II. Which features matter?

- Doesn't matter if its classical ML or deep learning neural networks. Going to touch on deep learning and what powers large language models later. 

# Act III. Why we regularize our features and how it works



# Act IV. How we choose features honestly and tune a model



# Act V. What is a neural network, really?
- Deep learning just means multi layered neural networks 
- Talk about how the hidden states explain different parts of the the output space
- backpropagation  
- Show example of text with autograd 
- A neural network = a parametrized mathematical function
- Autograd answers = How did every parameter contribute to the error? 
- Gradient descent then answers = How should every parameter change to reduce that error 

Make connection from neural network to large language model from this thing called 

The **attention mechanism**
—specifically a variation called "self-attention" introduced by Google researchers in the 2017 paper _"Attention Is All You Need"_—is the exact breakthrough that made Large Language Models (LLMs) possible.

Before this mechanism, scaling up neural networks to handle massive amounts of text was practically impossible due to how earlier architectures processed data.
Before 2017, one of the dominant architectures for language were **Recurrent Neural Networks (RNNs)** They processed text much like a human reads: sequentially, one word at a time.

**entire sequence of text all at once**.

- show the GPT 2 paper and the exact phrase that says this. 


# Act VI. A CPSC GPT
- Attention, how tokens attend to other tokens.
	- models are causual, each token attends to itself + previous tokens
	- **Context size = the maximum number of tokens the model can condition on at once.**
- Languages where you can convey as much information as possible in as little words is better for LLM's becuase of this attention. 
- How with other languages, where you need more characters to represetnt the same thing, you increase the amount of tokens that perceptro has to attent do AND there is more training data for the tokenizers in english than other languages (internet - Common Crawl [dataset](https://commoncrawl.github.io/cc-crawl-statistics/plots/languages) is 41% intnert, it has 10  ==Petabytes== (over 300 billion web pages))
- This was the problem a few years ago - show how こんにちは is 6 tokens with gpt2 but only 1 token with gpt-4- 
- Rerun the model with CPSc data 
- Talk about the stack: 
  - Inference:
    
    - the model is already trained
    - you feed in text and it predicts the next token or answers
    - this is the “use the model” phase
- Training / learning:
    
    - the model starts with random weights
    - it sees lots of data
    - it computes loss, backpropagates gradients, and updates weights
- softmax function is considered a generalization of the standard logistic (sigmoid) function




# Act VII. Unification





Act VI. What we did with gpt_train.py 
- Attention, how tokens attend to other tokens.
	- models are causual, each token attends to itself + previous tokens
	- **Context size = the maximum number of tokens the model can condition on at once.**
- Languages where you can convey as much information as possible in as little words is better for LLM's becuase of this attention. 
- How with other languages, where you need more characters to represetnt the same thing, you increase the amount of tokens that perceptro has to attent do AND there is more training data for the tokenizers in english than other languages (internet - Common Crawl [dataset](https://commoncrawl.github.io/cc-crawl-statistics/plots/languages) is 41% intnert, it has 10  ==Petabytes== (over 300 billion web pages))
- This was the problem a few years ago - show how こんにちは is 6 tokens with gpt2 but only 1 token with gpt-4- 
- Rerun the model with CPSc data 
- Talk about the stack: 
  - Inference:
    
    - the model is already trained
    - you feed in text and it predicts the next token or answers
    - this is the “use the model” phase
- Training / learning:
    
    - the model starts with random weights
    - it sees lots of data
    - it computes loss, backpropagates gradients, and updates weights
- softmax function is considered a generalization of the standard logistic (sigmoid) function


Act VII: Unification 















Act I: How text becomes geometry and machine readable

Act II. Which features matter?

Act III. Why we regularize our features and how it works

Act IV. How we choose features honestly and tune a model

Act V. What is a neural network, really?

Act VI. A CPSC GPT

Act VII. Unification

With LLMs, there are two different things:

- Inference:
    
    - the model is already trained
    - you feed in text and it predicts the next token or answers
    - this is the “use the model” phase
- Training / learning:
    
    - the model starts with random weights
    - it sees lots of data
    - it computes loss, backpropagates gradients, and updates weights

Conceptually, vocab size is the size of the model's _alphabet of tokens_