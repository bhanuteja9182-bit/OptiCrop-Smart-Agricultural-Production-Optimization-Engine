<<<<<<< HEAD
# 🌾 OptiCrop — Smart Agricultural Production Optimization Engine

OptiCrop is a machine-learning-powered web application that recommends the
most suitable crop to grow given a set of soil and climate readings —
Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH,
and rainfall. It is built for farmers, agricultural researchers, and
agribusinesses who want a fast, data-driven second opinion on crop
selection.

**Live demo flow:** enter soil/climate readings → get an instant crop
recommendation with a confidence score and the next-best alternatives.

---

## Table of contents

- [Problem statement](#problem-statement)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Entity Relationship Diagram](#entity-relationship-diagram)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Setup & run locally](#setup--run-locally)
- [API reference](#api-reference)
- [Screenshots](#screenshots)
- [Future improvements](#future-improvements)

---

## Problem statement

Farmers often struggle to choose the right crop for their land because soil
nutrient levels and climate conditions vary widely from field to field.
Poor crop selection leads to reduced yield, wasted resources, and financial
loss. OptiCrop addresses this by applying supervised machine learning to
historical agricultural data, learning the relationship between soil/climate
conditions and the crops that succeed under them, and surfacing that
knowledge through a simple web interface.

## Tech stack

| Layer            | Technology                                      |
|-------------------|--------------------------------------------------|
| Language           | Python 3.11                                      |
| Data handling      | NumPy, Pandas                                    |
| Visualization      | Matplotlib, Seaborn (fivethirtyeight style)      |
| Machine learning    | scikit-learn — Logistic Regression, K-Nearest Neighbors, K-Means Clustering |
| Model persistence  | joblib                                           |
| Backend            | Flask                                            |
| Frontend           | HTML5, CSS3, vanilla JavaScript                  |
| IDE / environment  | PyCharm, Anaconda Navigator                      |

## Project structure

```
OptiCrop/
├── app.py                      # Flask backend + /predict API
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py     # builds the crop dataset
│   └── Crop_recommendation.csv # 2200 rows x 22 crop varieties
├── notebooks/
│   ├── model_training.py       # EDA + preprocessing + model training
│   └── plots/                  # generated EDA & evaluation charts
├── models/
│   ├── crop_model.pkl          # trained Logistic Regression model
│   ├── scaler.pkl              # StandardScaler used at inference
│   ├── label_encoder.pkl       # crop label encoder/decoder
│   └── model_info.txt          # which model was selected + accuracy
├── templates/
│   └── index.html              # single-page UI
└── static/
    ├── css/style.css
    └── js/app.js
```

## Entity Relationship Diagram

The system's data model separates users, submitted soil readings, crops,
training datasets, trained models, predictions, and generated reports into
seven linked entities (`User`, `SoilData`, `Crop`, `Dataset`, `MLModel`,
`Prediction`, `Report`), each with its own primary key and foreign-key
relationships back to the records that produced it. This keeps the schema
normalized: a user can submit many soil records, each soil record produces
one prediction, and each prediction can generate multiple downstream
reports. See the ER diagram image in your SkillWallet workspace resources
for the full visual.

## Dataset

`data/Crop_recommendation.csv` contains 2,200 records across 22 crop
varieties (rice, maize, chickpea, kidney beans, pigeon peas, moth beans,
mung bean, black gram, lentil, pomegranate, banana, mango, grapes,
watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, and
coffee), with 100 balanced samples per crop and seven numeric features:
`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

> This repository generates a statistically representative version of the
> dataset via `data/generate_dataset.py`. If you have access to the original
> Kaggle dataset, drop your `Crop_recommendation.csv` into `data/` to
> replace it — the rest of the pipeline works unchanged since the schema is
> identical.

## Methodology

**Epic 2 — Data collection & analysis**
- Loaded and explored the dataset (`.head()`, `.describe()`, null checks)
- Univariate analysis: distribution plots per feature
- Bivariate analysis: humidity vs. crop label scatter plot
- Multivariate analysis: feature-mean bar chart + correlation heatmap

**Epic 3 — Data pre-processing**
- Verified there were no missing values
- Treated outliers per feature using the IQR method
- Split features (`X`) from the target label (`y`)
- 80/20 stratified train-test split

**Epic 4 — Model building**
- **K-Means Clustering** on the scaled features to explore natural
  groupings of agricultural conditions (used for pattern analysis, not for
  the final prediction)
- **Logistic Regression** trained as the primary classifier
- **K-Nearest Neighbors** trained as a comparison model
- Evaluated both with accuracy, precision/recall/F1, and confusion matrices
- The best-performing model (Logistic Regression) was persisted with
  `joblib` for deployment

**Epic 5 — Application building**
- Flask backend loads the trained model, scaler, and label encoder
- `/predict` endpoint accepts the 7 readings as JSON and returns the top
  crop plus two runner-up matches with confidence scores
- Responsive HTML/CSS/JS frontend for entering readings and viewing results

## Results

| Model                | Accuracy |
|-----------------------|----------|
| Logistic Regression   | **96.8%** |
| K-Nearest Neighbors   | 95.5%    |

Logistic Regression was selected as the production model. Full
classification reports and confusion matrices are saved to
`notebooks/plots/` after running the training script.

## Setup & run locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd OptiCrop

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Regenerate the dataset
python data/generate_dataset.py

# 5. (Optional) Retrain the model — pre-trained artifacts are already in models/
python notebooks/model_training.py

# 6. Run the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## API reference

**POST** `/predict`

Request body:
```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.8, "humidity": 82.0,
  "ph": 6.5, "rainfall": 202.9
}
```

Response:
```json
{
  "best_crop": "rice",
  "confidence": 67.6,
  "info": "Thrives in flooded fields with high humidity and warm temperatures.",
  "alternatives": [
    { "crop": "jute", "confidence": 31.0 },
    { "crop": "coffee", "confidence": 1.2 }
  ]
}
```

## Screenshots

Run the app locally and visit `/` to see the live interface — a soil-test
form on the left and a live recommendation panel (crop name, confidence
dial, and alternative matches) on the right. EDA charts generated during
training are saved under `notebooks/plots/`.

## Future improvements

- Swap the synthetic dataset for the original sourced dataset for
  production-grade accuracy
- Add user accounts so soil readings and past predictions are saved
  (matches the `User` → `SoilData` → `Prediction` → `Report` schema)
- Deploy to a cloud host (Render, Railway, or AWS) for a public demo link
- Add a PDF report export for each prediction

---

Built as part of an internship project — OptiCrop: Smart Agricultural
Production Optimization Engine.
=======
# OptiCrop-Smart-Agricultural-Production-Optimization-Engine
An AI-powered decision engine designed to optimize agricultural production by analyzing soil, weather, and crop data to maximize yield and efficiency.
>>>>>>> fe72082ef8beaf38c9f32e76b2dd1b350615721e
