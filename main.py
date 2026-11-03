import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

#Cargar dataset
df = pd.read_csv("500_Reddit_mensajes.csv")

#Lipieza nulos (no debería haber)
df = df.dropna(subset=["Mensaje", "Label"])

X = df["Mensaje"]
y = df["Label"]

print(f"Dimensiones del dataset cargado: {df.shape}")

#Partición (Train / Dev / Test)
#Train (80%) Test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train_full Train (80%) Dev (20%)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print("Tamaños de las particiones")
print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

#Vectorización baseline - TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)

#Algoritmo baseline - RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_vec, y_train)

print("-"*50)

#Evaluación sobre Dev
preds_dev = model.predict(X_dev_vec)
print(f"Macro F1-Score - Dev: {f1_score(y_dev, preds_dev, average='macro'):.4f}")

#Evaluación sobre Test
preds_test = model.predict(X_test_vec)
print(f"Macro F1-Score - Test: {f1_score(y_test, preds_test, average='macro'):.4f}")
print("Reporte de clasificación Test")
print(classification_report(y_test, preds_test))