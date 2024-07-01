# Of Machines and Men: Probing Neural Networks for Attraction Agreement with Psycholinguistic Data

## Theoretical background

Most of the interpretability studies focus on the problems, such as if models encode information or if models prefer grammatical sentences to ungrammatical. Such studies do not examine whether the models demonstrate the patterns similar to humans and whether they are sensible to the interfering humans' grammaticality judgements of phenomena. Such studies do not examine how similarly humans respond to tasks. It isn't clear, how exactly to compare human and models results, which is the main obstacle for potential studies. 

We probe BERT and GPT models on the syntactic phenomenon of agreement attraction on the psycholinguistic data with syncretism [1]. We suggest a new way of comparing models' and humans' response via statistical testing. We show that the models behave similarly to humans while GPT is more aligned with human responses than BERT. 


## Structure of the repository
- `attention`: this folder contains the code for our experiments with self-attention in ruBERT and ruGPT and statistical analyses of attention representations (see Section 4.2 in the paper)
- `perplexity`: this folder includes the code for the experiments with compatability scores and pseudoperplexity (see Section 4.1)
- `stats`: this folder includes code for statistical models to compare models' scores and humans' reading time in similar way

## References
[1] Slioussar, N. (2018). [Forms and features: The role of syncretism in number agreement attraction](https://www.sciencedirect.com/science/article/pii/S0749596X18300305). Journal of Memory and Language, 101, 51-63

## How to Cite
TBA
