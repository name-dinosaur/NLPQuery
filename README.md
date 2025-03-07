# NLPQuery
find most prob document in relation to user input words

OS
Windows
Python 3.12.6

SETUP
-Project folder should contain data folder with 3 pdfs, stanfordcorenlp and NLPQuery.py
  your folder should look like
  ![image](https://github.com/user-attachments/assets/04a23a01-676a-493a-88fd-570baed88e64)

-Before starting program navigate to stanfordcorenlp folder open terminal and run
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
-In terminal where folder .py folder is located run
  pip install pdfminer.six nltk

Run Program.

FURTHER CURATIONS would involve adding weights based on word importance (TF IDF), better handling of sparsity(backoff or something more complex), normalization (adjusting for documents of different sizes)

NOTE 
Theres two probs. One use regular prob where i added the log of them instead of multiplying regular probs so i didn't lose decimals since some probs were really small. Second used 2-ngram prob. Both used laplace smoothing.

