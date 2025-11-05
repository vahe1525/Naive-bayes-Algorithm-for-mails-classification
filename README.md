# 📧 Naive Bayes Spam Filter (Python)

This project implements a **Naive Bayes classifier** to automatically detect **spam emails** based on their subject, body, and sender.  
The goal is to **train a probabilistic model** that learns which words are more common in spam vs. legitimate (ham) emails.

---

## 🎯 Problem Description

The task is to classify an incoming email as either:
- **Spam** – unwanted or promotional content  
- **Ham** – legitimate, normal messages  

Each email is represented by:
- **Subject**
- **Body**
- **Sender**
- **Label** (*spam* or *ham*)

The model learns from a labeled dataset (`ttt.csv`) and predicts new email labels.

---

## 🧠 Algorithm: Naive Bayes Classifier

The **Naive Bayes** approach assumes that all words are **conditionally independent** given the class label.

It calculates:
\[
P(\text{class} | \text{email}) \propto P(\text{class}) \times \prod_{i=1}^{n} P(\text{word}_i | \text{class})
\]

### Steps
1. **Preprocessing** – Lowercasing, removing symbols, and tokenizing words  
2. **Training** – Counting word frequencies for each class (spam/ham)  
3. **Prediction** – Applying Bayes’ rule using probabilities with Laplace smoothing  
4. **Evaluation** – Measuring accuracy, precision, recall, and F1-score  

---

## 🧩 Dataset Format (`Emaildataset.csv`)

The dataset should contain columns:

| Subject | Body | Sender | Label |
|----------|------|--------|--------|
| Win a free iPhone! | Click here to claim your prize | promo@offers.com | spam |
| Meeting tomorrow | Don’t forget our meeting | manager@company.com | ham |
| ... | ... | ... | ... |

> Example: First 15 rows used for **training**, remaining for **testing**

---

## ⚙️ Key Classes

### **Email**
Represents an email message with:
- `subject`, `body`, `sender`, and `label`
- `full_text()` combines subject and body

### **NaiveBayesSpamFilter**
Implements the full algorithm:
- `train(emails)` → learns from labeled data  
- `predict(email)` → predicts `spam` or `ham`  
- `evaluate(test_data)` → computes Accuracy, Precision, Recall, and F1 Score  

---

## 📈 Evaluation Metrics

The model calculates:
| Metric | Formula | Meaning |
|---------|----------|----------|
| **Accuracy** | (TP + TN) / All | Overall correctness |
| **Precision** | TP / (TP + FP) | How many predicted spam are actually spam |
| **Recall** | TP / (TP + FN) | How many real spam were correctly found |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced performance measure |

---

## 🧾 Example Output
```bash

Model is Trained

--- ՄՈԴԵԼԻ ԳՆԱՀԱՏՄԱՆ ԱՐԴՅՈՒՆՔՆԵՐ ---
Accuracy: 93.33%
Precision: 90.00%
Recall: 100.00%
F1-Score: 94.74%

new predictions
New email predicted as: spam

```


