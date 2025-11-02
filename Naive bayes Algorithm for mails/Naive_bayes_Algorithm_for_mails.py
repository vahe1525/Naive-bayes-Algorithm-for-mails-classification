import re
import math
from collections import defaultdict
import pandas as pd

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

    def train(self, data):
       for text, label in data:
            words = self.preprocess(text)

            self.class_counts[label] += 1

            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)

       self.trained = True
 
    def predict(self, text: str) -> str: #this means that function takes one argument which is string and also returns a string  
        words = self.preprocess(text)

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

        for text, true_label in test_data:
            predicted_label = self.predict(text)
            
            if predicted_label == 'spam' and true_label == 'spam':
                TP += 1
            elif predicted_label == 'spam' and true_label == 'ham':
                FP += 1
            elif predicted_label == 'ham' and true_label == 'ham':
                TN += 1
            elif predicted_label == 'ham' and true_label == 'spam':
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



df = pd.read_csv("dataset.csv", encoding="utf-8")

# Train/Test բաժանում
train_df = df.iloc[:80]
test_df = df.iloc[80:]

train_data = list(zip(train_df["Text"], train_df["Label"]))
test_data = list(zip(test_df["Text"], test_df["Label"]))


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



#new object prediction
text1 = "hello, how are you baby"
text2 = "Get free gifts from Temu"

print("new predictions")
res1 = bayesModel.predict(text1)
print(f"this is email text -- : {text1} : and its a {res1} class mail")
res1 = bayesModel.predict(text2)
print(f"this is email text -- : {text2} : and its a {res1} class mail")
