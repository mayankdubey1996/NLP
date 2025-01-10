# NLP Preprocessing Pipeline

A comprehensive implementation of text preprocessing techniques using NLTK and spaCy libraries.

## 📝 Article
For a detailed explanation of the concepts and implementation, check out our [Medium article](https://medium.com/@mayankdubey1996/making-sense-of-text-data-a-business-guide-to-nlp-preprocessing-c81ea82bfa6d).

## 📂 Repository Structure

```bash
Week-1/

day1.py           # Text cleaning implementation
day2.py           # Tokenization exercises
day3.py           # Stop words removal
day4.py           # Stemming and lemmatization
data_preprocessing_pipeline.py    # Complete pipeline implementation
```

## 🎯 Mini Project
The [data_preprocessing_pipeline.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/data_preprocessing_pipeline.py) implements a complete text preprocessing pipeline with both NLTK and spaCy approaches. Key features:
- Text cleaning with regex
- Word and sentence tokenization
- Stop words removal
- Stemming and lemmatization
- Comparison between NLTK and spaCy results

## 📌 Weekly Deliverables Summary

| Day | Focus Area | Key Deliverables | Implementation |
|-----|------------|------------------|----------------|
| 1 | Text Cleaning | • Environment setup<br>• Basic text cleaning functions<br>• URL & punctuation removal | [day1.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day1.py) |
| 2 | Tokenization | • Word tokenization<br>• Sentence tokenization<br>• NLTK vs spaCy comparison | [day2.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day2.py) |
| 3 | Stop Words | • Stop words removal<br>• Multiple language support<br>• Comparison of approaches | [day3.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day3.py) |
| 4 | Normalization | • Stemming implementation<br>• Lemmatization<br>• Performance comparison | [day4.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day4.py) |
| 5 | Mini Project | • Complete pipeline<br>• BBC News dataset processing<br>• Comparative analysis | [data_preprocessing_pipeline.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/data_preprocessing_pipeline.py) |

## Expected Deliverables for Week 1

### 1. Individual Scripts:
- Text cleaning
- Tokenization 
- Stopword removal
- Stemming and lemmatization

### 2. Preprocessing Pipeline:
- A complete preprocessing pipeline function/script.

### 3. Preprocessed Dataset:
- A preprocessed dataset from the mini-project.

## 🛠️ Implementation Details

### Core Components:
1. **Text Cleaning** ([day1.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day1.py))
   - Lowercase conversion
   - Special character removal
   - Number removal
   - URL removal

2. **Tokenization** ([day2.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day2.py))
   - Word tokenization (NLTK & spaCy)
   - Sentence tokenization (NLTK & spaCy)

3. **Stop Words** ([day3.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day3.py))
   - NLTK stop words removal
   - spaCy stop words removal

4. **Word Normalization** ([day4.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/day4.py))
   - Porter Stemmer implementation
   - WordNet Lemmatizer
   - spaCy Lemmatization

## Weekly Plan

### Day 1: Introduction to Text Preprocessing

**Tasks:**
- Install and set up Python libraries: `nltk`, `spacy`, `textblob`.
- Learn the basics of text cleaning:
  - Lowercasing
  - Removing URLs, numbers, and punctuation
- Write a script to clean a sample text.

**Resources:**
- [Text Preprocessing Guide (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)
- Hands-On:
  - Implement a text cleaning script using regular expressions.

### Day 2: Tokenization

**Tasks:**
- Learn about word and sentence tokenization.
- Practice tokenization using NLTK and SpaCy.

**Resources:**
- [NLTK Documentation](https://www.nltk.org)
- [NLTK Tokenization](https://www.nltk.org/api/nltk.tokenize.html)
- [SpaCy Tokenization](https://spacy.io/usage/linguistic-features#tokenization)
- Video Tutorials:
  - [NLTK Implementation](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
  - [SpaCy Implementation](https://www.youtube.com/watch?v=_lR3RjvYvF4)

**Hands-On:**
- Tokenize sentences and words from a dataset or sample text.
- Compare results from NLTK and SpaCy.

### Day 3: Stop Words Removal

**Tasks:**
- Understand stopwords and their role in text preprocessing.
- Use NLTK and SpaCy to filter out stopwords.

**Resources:**
- [Understanding Stop Words](https://medium.com/@yashj302/stopwords-nlp-python-4aa57dc492af)
- [NLTK Stop Words](https://pythonspot.com/nltk-stop-words/)
- [SpaCy Stop Words](https://spacy.io/usage/rule-based-matching)

**Hands-On:**
- Remove stopwords from a dataset using NLTK and Spacy.

### Day 4: Stemming and Lemmatization

**Tasks:**
- Learn the difference between stemming and lemmatization.
- Implement stemming using NLTK and lemmatization using SpaCy.

**Resources:**
- [Stemming vs Lemmatization Guide](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)
- [SpaCy vs NLTK Comparison](https://devskrol.com/2021/11/28/spacy-stemming-vs-lemmatization/)

**Hands-On:**
- Apply stemming and lemmatization to the preprocessed text.
- Compare results and document observations.

### Day 5: Mini Project Implementation

**Tasks:**
- Combine all preprocessing steps into a complete pipeline
- Process the BBC News dataset
- Compare and evaluate different preprocessing approaches

**Implementation:**
- [data_preprocessing_pipeline.py](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/data_preprocessing_pipeline.py)
  - Combines text cleaning, tokenization, stop words removal, and normalization
  - Implements both NLTK and spaCy approaches
  - Processes a real-world dataset (BBC News)
  - Provides comparison between different preprocessing techniques

**Key Features of the Pipeline:**
1. **Modularity**
   - Each preprocessing step is implemented as a separate method
   - Easy to modify or extend individual components

2. **Comparison Framework**
   - Parallel implementation of NLTK and spaCy
   - Output columns for each preprocessing step
   - Ability to compare results between different approaches

3. **Performance Tracking**
   - Text length before and after cleaning
   - Processing time comparison
   - Quality assessment of different approaches

**Dataset:**
- BBC News Train dataset
- Contains news articles across different categories
- Perfect for testing various preprocessing techniques

**Results:**
- Clean, preprocessed text ready for NLP tasks
- Comparative analysis of NLTK vs spaCy approaches
- Insights into preprocessing impact on text data

**Resources:**
- [Complete Code Implementation](https://github.com/mayankdubey1996/NLP/blob/main/Week-1/data_preprocessing_pipeline.py)
- [Medium Article Explaining the Pipeline](https://medium.com/@mayankdubey1996/making-sense-of-text-data-a-business-guide-to-nlp-preprocessing-c81ea82bfa6d)

## 🔧 Setup

```bash
pip install nltk spacy pandas
python -m spacy download en_core_web_sm 
```

## 🚀 Usage

``` python
from preprocessing_pipeline import DataPreprocessingPipeline

# Initialize pipeline
pipeline = DataPreprocessingPipeline()

# Process your dataset
processed_df = pipeline.data_preprocessing_pipeline("your_dataset.csv")
```

## 📊 Output
The pipeline creates multiple columns for each preprocessing step, allowing you to compare different approaches and choose the most suitable one for your needs.