# Financial Sentiment Analysis

An NLP coursework project at the University of Tübingen (summer 2025) that classifies short financial texts as bearish, bullish or neutral. It compares a conventional text-classification baseline with a model using financial-domain language representations.

## Approach

- TF-IDF features plus financial/text features, classified with logistic regression.
- Pooled FinBERT embeddings plus engineered features, classified with a multilayer perceptron. FinBERT is used as an embedding extractor, not fine-tuned end to end.
- Evaluation with accuracy, macro F1, class-level reports and confusion matrices.

## Start here

Read the [report](Assignment5_Report_A_B.pdf) for the project write-up, or open the [notebook](Assignment5_Notebook_A_B.ipynb) for preprocessing, training and saved results.

**Stack:** Python, scikit-learn, PyTorch, Hugging Face Transformers/Datasets, NLTK and pandas.

## Reproduction notes

The notebook loads the `zeroshot/twitter-financial-news-sentiment` dataset and `ProsusAI/finbert` from Hugging Face. It requires the packages in its imports, NLTK resources and network access for downloads. If loading FinBERT fails, the code falls back to DistilBERT; check which model actually loaded before interpreting a rerun. This repository is an experiment in text classification, not a trading signal or investment model.
