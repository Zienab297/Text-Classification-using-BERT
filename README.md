# E-commerce Product Classification with BERT

This notebook demonstrates three different approaches to classifying e-commerce product descriptions into four categories: **Books**, **Clothing & Accessories**, **Electronics**, and **Household**.

## Dataset Overview
The dataset contains over 50,000 product descriptions. Preprocessing included removing null values, dropping duplicates, and label encoding the categories.

## Experiments & Methodology

### 1. Fine-tuned BERT Head
- **Model**: `bert-base-uncased` with a fresh classification head.
- **Approach**: The BERT backbone was frozen, and only the final classification layer was trained for 3 epochs using the AdamW optimizer.
- **Result**: Achieved high baseline performance by learning specific class mapping from labeled data.

### 2. Logistic Regression on Frozen BERT Embeddings
- **Model**: Scikit-learn `LogisticRegression` using BERT [CLS] token embeddings as features.
- **Approach**: Extracted 768-dimensional embeddings for all descriptions and trained a multi-class logistic regression model.
- **Result**: This method achieved the **highest overall accuracy**, proving the linear separability of BERT's pre-trained features.

### 3. Zero-Shot Learning (Cosine Similarity)
- **Model**: Frozen `bert-base-uncased` without any supervised training.
- **Approach**: Created optimized semantic 'anchor' descriptions for each class. Classification was performed by calculating the cosine similarity between product embeddings and category anchor embeddings.
- **Result**: Through iterative refinement of the category descriptions (prompt engineering), this baseline reached a functional performance level without requiring labeled training data.

## Final Performance Summary
| Model | Accuracy |
| :--- | :--- |
| **Logistic Regression (Frozen BERT)** | **92.86%** |
| **Fine-tuned BERT Head** | **85.79%** |
| **Optimized Zero-Shot** | **65.71%** |

## Data Integrity
All experiments utilized a stratified 80/20 train-test split. The validation set remained strictly unseen during the training of supervised models to ensure the results are a reliable measure of performance.
