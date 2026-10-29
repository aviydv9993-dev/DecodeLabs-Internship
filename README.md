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

- # Project 3: AI Recommendation Logic - Tech Stack Recommender

**DecodeLabs | Artificial Intelligence Industrial Training Kit | Batch 2026**

A content-based recommendation engine that maps a user's skills to the most relevant
tech career paths. It uses **TF-IDF weighting** and **cosine similarity**, both
implemented from scratch with NumPy, and returns the **Top 3** matching job roles.

---

## Goal

Create a simple recommendation system based on user preferences: take user input,
match preferences using similarity logic, and display the recommended items.

## How it works (IPO + 4-step ranking pipeline)

| Step | Stage     | What happens                                                          |
|------|-----------|-----------------------------------------------------------------------|
| 1    | Ingestion | Capture at least 3 skills, normalise wording, reject unknown skills   |
| 2    | Scoring   | Cosine similarity between the user vector and every job-role vector   |
| 3    | Sorting   | Rank roles by score, highest first                                    |
| 4    | Filtering | Truncate to the Top-N list (default 3) to avoid choice overload       |

### The maths

- **TF** = count of term in document / total terms in document
- **IDF** = ln((N + 1) / (df + 1)) + 1
  (smoothed version of ln(N / df), so a skill present in every role never drops to weight 0)
- **Weight** = TF x IDF, so rare, descriptive skills (e.g. `terraform`) count for more than
  common ones (e.g. `python`)
- **Cosine similarity** = (A . B) / (||A|| ||B||). It compares direction, not size, so a role
  with a long skill list is not unfairly penalised. Scores sit between 0 and 1 because
  TF-IDF values are never negative.

## Files

```
project3/
├── tech_stack_recommender.py   # the engine and CLI
├── raw_skills.csv              # dataset: 20 job roles and their skills
└── README.md
```

## Setup

```bash
pip install numpy
pip install scikit-learn      # optional, only for --verify
```

## Run

```bash
# Interactive onboarding survey
python tech_stack_recommender.py

# Pass skills directly (quote multi-word skills)
python tech_stack_recommender.py -s python "cloud computing" automation

# Show the top 5 instead of 3
python tech_stack_recommender.py -s python sql ml -n 5

# Prove the from-scratch maths equals scikit-learn's
python tech_stack_recommender.py --verify
```

### Sample output

```
Skills used : python, cloud, automation

1. Site Reliability Engineer  -  55.5% match
   You have : python, automation, cloud
   To learn : linux, kubernetes, monitoring, ci/cd, networking

2. Cloud Engineer  -  55.2% match
   You have : cloud, automation, python
   To learn : aws, azure, terraform, linux, networking

3. Cloud Architect  -  25.9% match
   You have : cloud
   To learn : aws, azure, networking, security, terraform, kubernetes, system design
```

## Features beyond the brief

- **From-scratch TF-IDF and cosine similarity**, checked against scikit-learn by `--verify`
  (max difference about 1e-16).
- **Shared vocabulary guard:** aliases map user wording onto dataset wording
  (`ml` -> `machine learning`, `cloud computing` -> `cloud`, `k8s` -> `kubernetes`), because
  mismatched naming makes the similarity maths fail.
- **Cold-start bypass:** the interactive survey keeps asking until at least 3 recognised
  skills are given, and suggests fixes for typos (`pyton` -> did you mean `python`?).
- **Explainable results:** every recommendation shows which skills you already have and
  which to learn next, making it a mini learning roadmap.
- **Edge cases handled:** duplicate skills, empty input, zero vectors (no divide-by-zero),
  and deterministic tie-breaking.

## Using your own dataset

Replace `raw_skills.csv` with your own file, or pass `--data other.csv`. Format:

```csv
role,skills
Data Scientist,python;sql;machine learning;statistics
```

If your instructor provides their own `raw_skills.csv`, drop it in and the code works
unchanged, as long as it has `role` and `skills` columns with skills separated by `;`.
Extend `ALIASES` in the script if their skill names differ from mine.

## Limitations

- Content-based filtering only recommends roles whose skills overlap with the input, so it
  cannot suggest surprising options the way collaborative filtering can.
- The dataset is small and hand-written; real results depend on the quality of the skill lists.
- Skills outside the dataset vocabulary are ignored.

## Ideas to extend

- Let users rate skills (1-5) instead of just listing them.
- Add a similarity score between roles ("roles similar to DevOps Engineer").
- Add a trending-roles fallback for users with no skills at all.
- Build a Streamlit or Flask front end.
- Compare against Jaccard similarity to see why TF-IDF + cosine ranks better.



## Author
Your Name - Avinash,DecodeLabs AI Intern, Batch 2026
