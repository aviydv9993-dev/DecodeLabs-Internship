# Rule-Based AI Chatbot ("Byte")

**DecodeLabs Industrial Training Kit - Artificial Intelligence, Project 1 (Batch 2026)**

A Python command-line chatbot that replies to predefined user inputs using rules and dictionaries. No machine learning is used. Every reply is traceable from input to logic to output.

## Features
- Continuous `while True` loop that runs until the user types an exit command
- Input sanitization: lowercase, trimmed spaces, punctuation removed
- Knowledge base: dictionary with 9 intents (greeting, how are you, name, help, joke, tip, thanks, time, date)
- Keyword lookup table (dictionary) instead of a long `if-elif` ladder
- Fallback reply for anything it does not understand
- Remembers your name (`my name is ...`)
- Random reply variety and live time/date answers
- Clean exit with `bye`, `exit` or `quit`

## How to run
```bash
python chatbot.py
```
Requires Python 3.9+. No external libraries needed.

## Example
```
Byte: Hi! I'm Byte. Type 'help' for ideas or 'bye' to exit.
You: Hello!
Byte: Hey there! I'm Byte, your coding study buddy.
You: my name is Riya
Byte: Nice to meet you, Riya!
You: tell me a joke
Byte: Why do programmers prefer dark mode? Because light attracts bugs!
You: bye
Byte: Goodbye, Riya! Keep coding.
```

## How it works (IPO model)
1. **Input:** read the message and clean it.
2. **Process:** look up the intent in a dictionary (O(1) lookup).
3. **Output:** print a reply from the knowledge base, or a fallback.

## Skills demonstrated
Control flow, loops, dictionaries, string handling, decision-making logic, basic AI concepts.


# Project 2: Data Classification Using AI

**DecodeLabs | Artificial Intelligence Industrial Training Kit | Batch 2026**

A supervised-learning pipeline that classifies iris flowers into three species
(*setosa*, *versicolor*, *virginica*) from four measurements, using
**K-Nearest Neighbors (KNN)** and scikit-learn.

---

## Goal

Build a basic classification model on a small dataset by completing the core
supervised-learning workflow: **load -> split -> train -> validate**.

## Pipeline (IPO Framework)

| Stage       | What happens                                                              |
|-------------|---------------------------------------------------------------------------|
| **Input**   | Load the Iris dataset, check for missing values and class balance         |
| **Process** | Stratified 80/20 split, StandardScaler, tune K with cross-validation, KNN |
| **Output**  | Accuracy, confusion matrix, precision / recall / F1, saved plots          |

## What makes this implementation different

- **No data leakage:** `StandardScaler` sits inside a scikit-learn `Pipeline`, so it is
  fitted on training data only and merely applied to test data.
- **Stratified split:** every species keeps the same proportion in train and test.
- **Data-driven K:** K is picked by 5-fold cross-validation on the training set, so the
  test set stays untouched until the final evaluation.
- **Algorithm benchmark:** KNN is compared with Logistic Regression, SVM and a
  Decision Tree on the same split.
- **Real predictions:** `predict_flower()` classifies new measurements and shows the
  neighbour vote shares per class.

## Dataset

| Property   | Value                                                   |
|------------|---------------------------------------------------------|
| Samples    | 150 (50 per class, balanced)                            |
| Classes    | 3: setosa, versicolor, virginica                        |
| Features   | sepal length, sepal width, petal length, petal width (cm) |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install scikit-learn pandas numpy matplotlib
```

## Run

```bash
python iris_classifier.py
```

Output is printed to the terminal and two plots are written to `outputs/`:

- `k_tuning.png`: error rate vs K (the "elbow" chart)
- `confusion_matrix.png`: actual vs predicted classes on the test set

## Project structure

```
.
├── iris_classifier.py   # full pipeline
├── README.md
└── outputs/
    ├── k_tuning.png
    └── confusion_matrix.png
```

## Key concepts

- **Supervised learning:** we supply labelled history and the machine derives the rules.
- **Feature scaling:** KNN is distance-based, so features must share a common scale.
- **Train/test split:** the test set simulates unseen data and is never used for tuning.
- **Choosing K:** too small overfits to noise, too large underfits and becomes generic.
- **Confusion matrix:** shows TP, FP, FN and TN, i.e. *which* mistakes are made.
- **F1 score:** the harmonic mean of precision and recall. It is more honest than raw
  accuracy, especially on imbalanced data.

## Results

Sample run (seed 42, 120 train / 30 test rows):

| Metric                    | Value |
|---------------------------|-------|
| Best K (5-fold CV)        | 5     |
| CV accuracy (train only)  | 0.967 |
| Test accuracy             | 0.933 |
| Test macro F1             | 0.933 |

The confusion matrix shows *setosa* and *versicolor* classified perfectly. The only
errors are 2 *virginica* flowers predicted as *versicolor*, the two species whose
measurements overlap. On a 30-row test set, one flower is worth about 3.3 accuracy
points, so small differences between models (KNN 0.933 vs SVM / Decision Tree 0.967)
are not statistically meaningful.

## Ideas to extend

- Test with completely new measurements via `predict_flower()`.
- Swap in another dataset (Wine, Breast Cancer) by changing `load_data()`.
- Try different distance metrics (`metric="manhattan"`) or `weights="distance"`.
- Add `GridSearchCV` to tune several hyperparameters at once.
- Next milestone: Deep Learning and CNNs for computer vision.


## Author
Your Name - Avinash,DecodeLabs AI Intern, Batch 2026
