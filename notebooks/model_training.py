"""
model_training.py
------------------
OptiCrop: Smart Agricultural Production Optimization Engine

Covers:
  Epic 2 - Data Collection and Analysis (read data, univariate/bivariate/
           multivariate analysis)
  Epic 3 - Data Pre-processing (null check, outlier handling, feature/
           label split, train-test split)
  Epic 4 - Model Building (K-Means Clustering, Logistic Regression,
           evaluation, save best model)

Run:  python model_training.py
Outputs plots into notebooks/plots/ and the trained model + label encoder
+ scaler into ../models/
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Crop_recommendation.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ---------------------------------------------------------------------------
# Epic 2: Data Collection and Analysis
# ---------------------------------------------------------------------------
def read_dataset():
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print(df.head())
    return df


def univariate_analysis(df):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    fig.delaxes(axes[-1])
    plt.suptitle("Distribution of agricultural conditions")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "univariate_distributions.png"), dpi=120)
    plt.close()


def bivariate_analysis(df):
    plt.figure(figsize=(8, 10))
    sns.scatterplot(x=df["humidity"], y=df["label"])
    plt.xlabel("humidity")
    plt.ylabel("label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_humidity_vs_label.png"), dpi=120)
    plt.close()


def multivariate_analysis(df):
    plt.figure(figsize=(10, 5))
    df[FEATURES].mean().plot(kind="bar", color=sns.color_palette("tab10"))
    plt.title("Average agricultural condition values")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multivariate_feature_means.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation between agricultural features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    print(df[FEATURES].describe())


# ---------------------------------------------------------------------------
# Epic 3: Data Pre-processing
# ---------------------------------------------------------------------------
def check_nulls(df):
    nulls = df.isnull().sum()
    print("Null values per column:\n", nulls)
    return nulls


def handle_outliers(df):
    """Cap outliers using the IQR method for each numeric feature."""
    df_clean = df.copy()
    for col in FEATURES:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_clean[col] = df_clean[col].clip(lower, upper)
    return df_clean


def split_features_label(df):
    X = df[FEATURES]
    y = df["label"]
    return X, y


def train_test_splitting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Epic 4: Model Building
# ---------------------------------------------------------------------------
def run_kmeans_clustering(X_scaled, k=22):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    print(f"K-Means silhouette score (k={k}): {score:.3f}")
    return km, clusters


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    return acc


def save_confusion_matrix(model, X_test, y_test, encoder, name):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{name}.png"), dpi=120)
    plt.close()


def predict_best_crop(model, scaler, encoder, N, P, K, temperature, humidity, ph, rainfall):
    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES
    )
    sample_scaled = scaler.transform(sample)
    pred_encoded = model.predict(sample_scaled)[0]
    return encoder.inverse_transform([pred_encoded])[0]


def main():
    df = read_dataset()
    check_nulls(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)

    df_clean = handle_outliers(df)
    X, y = split_features_label(df_clean)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_splitting(X, y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # K-Means clustering (unsupervised exploration of agricultural patterns)
    run_kmeans_clustering(X_train_scaled, k=len(encoder.classes_))

    # Logistic Regression
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    log_reg_acc = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
    save_confusion_matrix(log_reg, X_test_scaled, y_test, encoder, "LogisticRegression")

    # KNN (extra model for comparison, as referenced in Business Requirements)
    knn = train_knn(X_train_scaled, y_train)
    knn_acc = evaluate_model(knn, X_test_scaled, y_test, "K-Nearest Neighbors")
    save_confusion_matrix(knn, X_test_scaled, y_test, encoder, "KNN")

    # Pick and persist the best-performing model
    best_model, best_name, best_acc = (
        (log_reg, "LogisticRegression", log_reg_acc)
        if log_reg_acc >= knn_acc
        else (knn, "KNN", knn_acc)
    )
    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "model_info.txt"), "w") as f:
        f.write(f"Best model: {best_name}\nAccuracy: {best_acc:.4f}\n")

    # Sample prediction demo
    example = predict_best_crop(
        best_model, scaler, encoder,
        N=90, P=42, K=43, temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.9
    )
    print(f"\nSample prediction for given soil/climate parameters -> {example}")


if __name__ == "__main__":
    main()
