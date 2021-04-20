# DrQA Adaptive Document Retriever
This is an implementation of [Adaptive Document Retrieval for Deep Question Answering](https://github.com/bernhard2202/adaptive-ir-for-qa) **only** for [DrQA](https://github.com/facebookresearch/DrQA) Document Retriever on [PubMedQA Dataset](https://github.com/pubmedqa/pubmedqa).

## Usage

- [Install DrQA](https://github.com/facebookresearch/DrQA#installing-drqa) as stated in DrQA repo documentation.
- [Build the document database](https://github.com/facebookresearch/DrQA/tree/master/scripts/retriever).
- [Build the TF-IDF N-grams](https://github.com/facebookresearch/DrQA/tree/master/scripts/retriever#building-the-tf-idf-n-grams).
- [Convert the dataset to the correct format](https://github.com/facebookresearch/DrQA#format-a) in order to generate training data to make the retriever adaptive.
- Generate training data by running `python scripts/retriever/eval.py /path/to/txtdataset --model /path/to/tfidf --doc-db /path/to/db --n-docs 25`, the output is in the file `training_adaptive.csv`
- Train the adaptive model by running `python adaptive_model_training.py`. The model will be saved as `adaptive_model.sav`
- Move `adaptive_model.sav` to `drqa/retriever`.
- Initialize the retriever pipeline by running `python scripts/retriever/pipeline.py --model /path/to/tfidf`. 
- Ask any question with `k=25` and the script will give different number of top documents according to your query :
```
>>> process('Is breast cancer dangerous ?', k=25)
+------+----------+-----------+
| Rank |  Doc Id  | Doc Score |
+------+----------+-----------+
|  1   | 15050326 |  0.060087 |
|  2   | 17598882 |  0.056639 |
|  3   | 14692023 |  0.050878 |
+------+----------+-----------+
```