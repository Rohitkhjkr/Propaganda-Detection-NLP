# Propaganda-Detection-NLP
NLP project for propaganda technique classification and span identification using TF-IDF, Logistic Regression and DistilBERT.

# Overview
This project investigates automatic detection of propaganda in text through two related tasks:
1. **Propaganda Technique Classification** – classifying a given propaganda span into one of eight propaganda techniques.
2. **Propaganda Span Identification** – identifying the propaganda spans within a sentence and classifying them using BIO tagging.
The project compares traditional TF-IDF-based machine learning with transformer-based DistilBERT models.

# Dataset
The dataset contains propaganda-labelled sentences in TSV format, with propaganda spans marked using `<BOS>` and `<EOS>` tags.
For Task 1, non-propaganda examples were removed because the objective was to classify propaganda techniques.
For Task 2, the complete sentence context was retained and BIO labels were generated:
- `B` – beginning of a propaganda span
- `I` – inside a propaganda span
- `O` – outside a propaganda span

# Methods

# Task 1A: TF-IDF + Logistic Regression
A traditional machine-learning baseline was developed using:
- TF-IDF features
- Unigrams and bigrams
- Maximum 10,000 features
- English stop-word removal
- Balanced class weights
- Logistic Regression

# Task 1B: DistilBERT
A `distilbert-base-uncased` transformer model was used to classify propaganda techniques from extracted spans. The model used contextual representations to capture semantic relationships that may not be represented by frequency-based TF-IDF features.

# Task 2A: Generic BIO Tagging
A pipeline approach was used in which propaganda sentences were first identified and then BIO tagging was applied to identify propaganda spans.

# Task 2B: Technique-Aware BIO Tagging
A DistilBERT token-classification approach was used to predict technique-specific BIO labels.
The low performance was associated with the large number of BIO labels, strong class imbalance and the limited dataset size.

The results show that DistilBERT performed better than the TF-IDF + Logistic Regression baseline for propaganda technique classification. For span identification, the generic BIO tagging pipeline achieved substantially higher span-level performance than the technique-aware approach.

# Preprocessing
The preprocessing workflow included:
- Extraction of propaganda spans using `<BOS>` and `<EOS>` markers
- Removal of missing spans
- Dataframe index resetting
- TF-IDF feature extraction
- DistilBERT tokenisation
- Padding and truncation
- BIO label generation
- Token-label alignment for transformer subwords

# Training
The transformer-based models used:
- DistilBERT
- AdamW optimisation
- Learning rate of `1e-5`
- Gradient clipping
- Learning-rate scheduling
- Validation during training
Different sequence lengths and batch sizes were used for classification and token-classification tasks because of their different context and memory requirements.

# Evaluation
Model performance was evaluated using:
- Macro-F1 for propaganda technique classification
- Span-Level F1 for propaganda span identification
- Confusion matrices
- BIO tag distribution analysis
- Token-level prediction error analysis

# Key Findings
- DistilBERT outperformed TF-IDF + Logistic Regression for technique classification.
- Contextual representations helped distinguish propaganda techniques with overlapping linguistic characteristics.
- Generic BIO tagging performed substantially better than technique-aware BIO tagging for span detection.
- Severe BIO class imbalance made token-level propaganda detection difficult.
- Limited dataset size increased the risk of weak generalisation for transformer-based models.

