import re
import math
from collections import defaultdict
import pandas as pd

class Email:
    def __init__(self, subject: str, body: str, sender: str, label: str = None):
        self.subject = subject
        self.body = body
        self.sender = sender
        self.label = label

    def full_text(self):
        return f"{self.subject} {self.body}"


class NaiveBayesSpamFilter:
    def __init__(self):
        self.word_counts = {
            "spam": defaultdict(int),
            "ham": defaultdict(int)
        }
        self.class_counts = {
            "spam": 0,
            "ham": 0
        }
        self.vocab = set()
        self.trained = False 

    def preprocess(self, text: str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        tokens = re.findall(r"\b[a-z]+\b", text)
        return tokens

    def train(self, emails):
       for email in emails:
            text = email.full_text()
            label = email.label
            
            words = self.preprocess(text)

            self.class_counts[label] += 1

            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)

       self.trained = True
 
    def predict(self, email: Email) -> str: #this means that function takes one argument which is string and also returns a string  
        words = self.preprocess(email.full_text())

        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)

        scores = {}

        for label in self.class_counts.keys():
            # prior = P(label)
            score = self.class_counts[label] / total_docs   # օրինակ 2/3 կամ 1/3

            total_words_in_class = sum(self.word_counts[label].values())  # spam→9, ham→5

            # multiplicative form
            for w in words:
                count = self.word_counts[label].get(w, 0)   # N(բառ,դաս)
                prob = (count + 1) / (total_words_in_class + vocab_size)
                score *= prob      # հենց սա ա առանց log-ի տարբերակը

            scores[label] = score

        # վերադարձնում ենք ամենամեծը
        return max(scores, key=scores.get)


    def evaluate(self, test_data):

        TP, FP, TN, FN = 0, 0, 0, 0 
        
        total_predictions = len(test_data)
        if total_predictions == 0:
            return None

        for email in test_data:
            predicted_label = self.predict(email)
            
            if predicted_label == 'spam' and email.label == 'spam':
                TP += 1
            elif predicted_label == 'spam' and email.label == 'ham':
                FP += 1
            elif predicted_label == 'ham' and email.label == 'ham':
                TN += 1
            elif predicted_label == 'ham' and email.label == 'spam':
                FN += 1
        
        accuracy = (TP + TN) / total_predictions
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score
        }



# df = pd.read_csv("ttt.csv", encoding="utf-8")
df = pd.read_csv("ttt.csv", encoding_errors="ignore")


# Train/Test բաժանում
train_df = df.iloc[:15]
test_df = df.iloc[15:]

# 📨 Build Email objects
train_data = [Email(row["Subject"], row["Body"], row["Sender"], row["Label"]) for _, row in train_df.iterrows()]
test_data = [Email(row["Subject"], row["Body"], row["Sender"], row["Label"]) for _, row in test_df.iterrows()]


bayesModel = NaiveBayesSpamFilter()
bayesModel.train(train_data)
print("Model is Trained")

evaluation_results = bayesModel.evaluate(test_data)

print("\n--- ՄՈԴԵԼԻ ԳՆԱՀԱՏՄԱՆ ԱՐԴՅՈՒՆՔՆԵՐ ---")
if evaluation_results:
    print(f"Accuracy:  {evaluation_results['Accuracy'] * 100:.2f}%")
    print(f"Precision: {evaluation_results['Precision'] * 100:.2f}%")
    print(f"Recall:    {evaluation_results['Recall'] * 100:.2f}%")
    print(f"F1-Score:  {evaluation_results['F1_Score'] * 100:.2f}%")

print("new predictions")
# ✉️ Try predictions
new_email = Email(
    "Special crypto reward!",
    "Claim your exclusive bonus now by joining our platform.",
    "crypto@offerhub.com"
)
print(f"\nNew email predicted as: {bayesModel.predict(new_email)}")