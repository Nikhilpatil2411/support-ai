import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("customer_support_tickets_200k.csv")

# select columns
df = df[["issue_description","category"]]

df = df.dropna()

# reduce dataset to 20000 rows (better learning)
df = df.sample(20000, random_state=42)

print("Records used:",len(df))

# clean text
def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]','',text)

    return text

df["clean_text"] = df["issue_description"].apply(clean_text)

X = df["clean_text"]
y = df["category"]

vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X_vec,
    y,
    test_size=0.20,
    random_state=42
)

model = LogisticRegression(max_iter=2000)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy:",accuracy)

joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model saved successfully")

