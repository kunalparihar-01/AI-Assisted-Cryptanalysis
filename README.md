# AI-Assisted Cryptanalysis of Classical Ciphers

An interactive cybersecurity and cryptography project that combines **statistical analysis, language scoring, machine learning, and cryptanalysis techniques** to analyze and recover plaintext from classical ciphers.

## 🌐 Live Demo

**[Launch AI Cryptanalysis Lab](https://ai-cryptanalysis.streamlit.app/)**

## 🐙 GitHub Repository

**[AI-Assisted-Cryptanalysis](https://github.com/kunalparihar-01/AI-Assisted-Cryptanalysis)**

---

## 📌 Project Overview

Classical ciphers are important for understanding the fundamental concepts of encryption and cryptanalysis. This project provides an interactive laboratory for experimenting with classical ciphers and demonstrates how statistical analysis and machine learning can assist in identifying and analyzing encrypted text.

### Features

- Encryption and decryption
- Automatic cipher detection
- Frequency analysis
- N-gram analysis
- Language scoring
- Machine-learning-based cipher classification
- Cryptanalysis and candidate ranking
- Interactive Streamlit interface

The project focuses on **educational cryptanalysis of classical ciphers** and is intended for learning, experimentation, and demonstration.

---

## 🔐 Supported Classical Ciphers

### Caesar Cipher

A substitution cipher in which each letter is shifted by a fixed number of positions in the alphabet.

Supports:
- Encryption
- Decryption
- Brute-force cryptanalysis
- Statistical/rule-based detection
- ML-based detection

### Vigenère Cipher

A polyalphabetic substitution cipher that uses a repeating keyword.

Supports:
- Encryption
- Decryption
- Key-length estimation
- Key recovery
- Language-score-based candidate ranking
- Kasiski and statistical analysis

### Monoalphabetic Substitution Cipher

A cipher in which each plaintext letter is consistently mapped to another letter.

Supports:
- Encryption
- Decryption
- Frequency-based analysis
- Simulated-annealing-based cryptanalysis
- Language-score-based candidate ranking

---

## 🤖 Machine Learning

The project includes a machine-learning classifier for identifying the cipher type.

### Dataset

| Property | Value |
|---|---:|
| Total samples | 2,400 |
| Caesar | 800 |
| Vigenère | 800 |
| Substitution | 800 |
| Features per sample | 48 |
| Number of classes | 3 |

### Test Performance

| Metric | Result |
|---|---:|
| Accuracy | **93.5%** |
| Macro Precision | **93.8%** |
| Macro Recall | **93.5%** |

### Per-Class F1 Score

| Cipher | F1 Score |
|---|---:|
| Caesar | 0.93 |
| Substitution | 0.92 |
| Vigenère | 0.96 |

The model uses statistical and linguistic features such as letter-frequency characteristics, index of coincidence, entropy, n-gram information, and related IC-profile features.

> **Note:** The reported metrics are based on the project's generated/test dataset and should be interpreted as evaluation results for this educational implementation, not as a guarantee of performance on arbitrary real-world ciphertext.

---

## 📊 Statistical & Frequency Analysis

The analysis module provides:

- Total number of letters
- Distinct letters
- Letter-frequency distribution
- Index of Coincidence (IC)
- Shannon entropy
- Chi-square statistics
- Common bigrams
- Common trigrams
- Language-score components

These measurements provide evidence that can assist cipher identification and cryptanalysis.

---

## 🧠 Cryptanalysis Pipeline

The workflow combines multiple signals rather than relying on a single technique.

```text
                Ciphertext
                    |
                    v
          Statistical Analysis
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
   Frequency       IC        Entropy/
    Analysis                 Chi-square
        |           |           |
        +-----------+-----------+
                    |
                    v
             Cipher Detection
              +-----+-----+
              |           |
              v           v
            Rules        ML
              |           |
              +-----+-----+
                    |
                    v
          Candidate Generation
                    |
                    v
           Language Scoring
                    |
                    v
            Candidate Ranking
                    |
                    v
          Recovered Plaintext
```

The candidate-ranking stage uses language evidence and cipher-detection confidence to reduce the chance of a statistically plausible but unreadable candidate being ranked first.

---

## 🧪 Testing

The project includes automated tests covering cipher implementations, statistical analysis, ML components, and regression cases.

### Current Test Result

**203 tests passed — 0 failed**

```text
203 passed
```

Regression coverage includes:
- Caesar detection and cryptanalysis
- Vigenère key recovery and ranking
- Substitution cryptanalysis
- Language scoring
- Statistical detection signals
- ML-related behavior

---

## 🖥️ Streamlit Application

The application provides an interactive web interface with:

- **Home**
- **Encrypt**
- **Decrypt**
- **Cryptanalysis**
- **Frequency Analysis**
- **ML Model**
- **About**

**[Launch the live application](https://ai-cryptanalysis.streamlit.app/)**

---

## 📁 Project Structure

```text
AI-Assisted-Cryptanalysis/
|
+-- analysis/
|   +-- __init__.py
|   +-- cipher_detection.py
|   +-- frequency.py
|   +-- language_score.py
|   +-- ngrams.py
|
+-- ciphers/
|   +-- __init__.py
|   +-- caesar.py
|   +-- ngram_data.py
|   +-- substitution.py
|   +-- vigenere.py
|
+-- data/
|   +-- cipher_model.joblib
|   +-- cipher_model_metrics.json
|   +-- sample_dataset.csv
|
+-- ml/
|   +-- __init__.py
|   +-- features.py
|   +-- predictor.py
|   +-- train.py
|
+-- tests/
|   +-- __init__.py
|   +-- test_analysis.py
|   +-- test_ciphers.py
|   +-- test_ml.py
|   +-- test_regression.py
|
+-- utils/
|   +-- __init__.py
|   +-- helpers.py
|
+-- app.py
+-- requirements.txt
+-- .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kunalparihar-01/AI-Assisted-Cryptanalysis.git
cd AI-Assisted-Cryptanalysis
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🧪 Running Tests

From the project root:

```bash
pytest
```

Expected result for the current project version:

```text
203 passed
```

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Pytest**
- Classical cryptography and cryptanalysis techniques
- Statistical language analysis
- N-gram analysis
- Machine learning

---

## 🎯 Educational Objectives

This project demonstrates how:

1. Classical encryption algorithms work.
2. Statistical properties can reveal information about ciphertext.
3. Frequency and n-gram analysis can support cryptanalysis.
4. Machine learning can classify different cipher types.
5. Multiple analytical signals can be combined into a cryptanalysis pipeline.
6. Automated testing can validate a cybersecurity application.

---

## ⚠️ Scope & Responsible Use

This project is designed for **educational and defensive cybersecurity research**.

It focuses on classical ciphers and controlled/synthetic test data. It should not be interpreted as a tool for breaking modern cryptographic systems or accessing protected information without authorization.

---

## 🚀 Future Improvements

Possible future work includes:

- Larger and more diverse language corpora
- Additional classical ciphers
- Improved multilingual language models
- More advanced ML classification
- Visualization of cryptanalysis search processes
- Additional benchmarking datasets
- Expanded automated regression testing

---

## 👨‍💻 Project

**AI-Assisted Cryptanalysis of Classical Ciphers Using Machine Learning and Statistical Analysis**

**GitHub:** https://github.com/kunalparihar-01/AI-Assisted-Cryptanalysis

**Live Demo:** https://ai-cryptanalysis.streamlit.app/
