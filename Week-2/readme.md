# Text Vectorization and Similarity Metrics

A comprehensive implementation of text vectorization techniques using Bag-of-Words (BoW) and TF-IDF, along with similarity metrics.

## 📂 Repository Structure

```bash
Week-2/
├── Data/
│   └── bbc-news-data.csv             # BBC News Dataset
├── bag_of_words.py                   # BoW implementation
├── tf_idf.py                         # TF-IDF implementation
├── similarity_matrix.py              # Similarity implementation
├── preprocessing_bow_tfidf.py        # Preprocessing for vectorization
└── keywords_extractor.py             # Keyword extraction implementation
```

## 🎯 Mini Project
The [keyword_extractor.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/keyword_extractor.py) implements a complete keyword extraction pipeline using TF-IDF. Key features:
- Text preprocessing and vectorization
- TF-IDF based keyword extraction
- Top-K keywords identification
- Results export to CSV

## 📌 Weekly Deliverables Summary

| Day | Focus Area | Key Deliverables | Implementation |
|-----|------------|------------------|----------------|
| 1 | Bag-of-Words | • BoW understanding<br>• Vector generation<br>• Vocabulary creation | [day1.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day1.py) |
| 2 | TF-IDF | • TF-IDF implementation<br>• Word importance scoring<br>• Vector comparison | [day2.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day2.py) |
| 3 | Preprocessing | • Text cleaning<br>• Tokenization<br>• Stop words removal | [day3.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day3.py) |
| 4 | Cosine Similarity | • Similarity calculation<br>• Document comparison<br>• Vector analysis | [day4.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day4.py) |
| 5-7 | Keyword Extractor | • Complete pipeline<br>• Movie plots analysis<br>• Keyword extraction | [keyword_extractor.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/keyword_extractor.py) |

## 🛠️ Implementation Details

### Core Components:
1. **Bag-of-Words** ([day1.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day1.py))
   - Vocabulary creation
   - Word count vectorization
   - Document vector generation

2. **TF-IDF** ([day2.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day2.py))
   - Term frequency calculation
   - Inverse document frequency
   - Final weight computation

3. **Text Preprocessing** ([day3.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day3.py))
   - Cleaning and normalization
   - Tokenization
   - Stop words removal

4. **Similarity Metrics** ([day4.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-2/day4.py))
   - Cosine similarity implementation
   - Document comparison
   - Similarity score calculation

## Weekly Plan
### Day 1: Bag-of-Words Basics
**Tasks:**

- Install and set up Python libraries: scikit-learn, pandas, numpy
- Learn the basics of Bag-of-Words:
   - Vocabulary creation
   - Word count vectors
   - Document representation
- Practice generating word vectors

**Resources:**

[Scikit-Learn Text Feature Extraction](https://scikit-learn.org/1.5/modules/feature_extraction.html)
[Bag of Words : Natural Language Processing](https://www.youtube.com/watch?v=irzVuSO8o4g)

#### Day 2: Introduction to TF-IDF
**Tasks:**

- Learn about TF-IDF components
- Understand term frequency and document frequency
- Practice implementing TF-IDF

**Resources:**

[Scikit-Learn TF-IDF Guide](https://scikit-learn.org/1.5/modules/feature_extraction.html)
[TFIDF : Data Science Concepts](https://www.youtube.com/watch?v=OymqCnh-APA&t=24s)

#### Day 3: Preprocessing for Vectorization
**Tasks:**

- Combine preprocessing with vectorization techniques
- Practice cleaning and preparing text for BoW/TF-IDF
- Apply techniques to BBC News Dataset

**Resources:**

[Kaggle - Text Preprocessing for NLP](https://www.kaggle.com/c/learn-ai-bbc/data)

**Hands-On:**

- Implement preprocessing pipeline for vectorization



#### Day 4: Similarity Metrics Implementation

**Tasks:**

- Learn about cosine similarity
- Implement document similarity calculation
- Practice comparing text vectors

**Resources:**

[Cosine Similarity, Clearly Explained!!!](https://www.youtube.com/watch?v=e9U0QAFbfLI&t=365s)

**Hands-On:**
- Calculate similarity between documents



#### Day 5-7: Mini Project - Keyword Extractor

**Tasks:**

- Implement a complete keyword extraction system
- Use TF-IDF for finding important words
- Process BBC News dataset
- Extract and evaluate keywords

Resources:

[Keyword Extraction with TF-IDF](https://www.youtube.com/watch?v=TBUpxFw8oIA)
Hands-On:

- Build end-to-end keyword extraction pipeline
- Evaluate results on news articles   

## 🔧 Setup

```bash
pip install scikit-learn pandas numpy
```

from keyword_extractor import KeywordExtractor

## 🚀 Usage

# Initialize extractor

``` python
extractor = KeywordExtractor()
```

# Extract keywords from text

```python
keywords = extractor.extract_keywords("your_text_here", top_k=10)
```

## 📊 Output
- The keyword extractor generates a CSV file containing:
   - Document ID/Title
   - Original text
   - Top K keywords
   - Keyword importance scores