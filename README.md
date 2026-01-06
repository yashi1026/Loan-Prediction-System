# Loan Prediction System for Banks

## 🧠 Problem Statement

Loans are a primary business for banks and financial institutions. They must analyze if an applicant can **repay the loan without defaulting**. Traditional manual verification is time-consuming and subjective.

This system uses historical loan data to **automate the decision-making process**, minimizing human workload and enabling **data-driven predictions**.

---

## 🎯 Goal of the Project

Dream Housing Finance deals with home loans across urban, semi-urban, and rural regions. The goal is to build a model that:

- Classifies applicants as **eligible (Approved)** or **ineligible (Rejected)** for a loan  
- Helps financial institutions quickly assess loan risk  
- Provides real-time predictions based on user inputs

---

## 📊 Theoretical Background

### 🔍 Loan Prediction as a Machine Learning Problem

Loan prediction is a **binary classification problem** where the model learns patterns from past loan decisions and predicts outcomes for new applications.  
Supervised learning algorithms like **Logistic Regression** and other classifiers are trained to distinguish between **approved** and **rejected** cases based on features such as income, credit history, and loan amount.

#### Key Concepts:

- **Exploratory Data Analysis (EDA):** Understand data distribution, feature relationships, outliers, and missing values.  
- **Feature Engineering:** Create new attributes like total income or EMI ratio to give the model more meaningful information.  
- **Model Training:** Use labeled historical data to train a classifier.  
- **Evaluation:** Assess model using accuracy and classification metrics.

---

## 📁 Project Structure

Loan-Prediction-System/
│
├── Exploratory Data Analysis.ipynb # Exploratory analysis and visualizations
├── Loan Prediction.ipynb # Model training pipeline
├── Model Building.ipynb # Model creation and evaluation
├── app.py # Streamlit web application
├── loan_model.pkl # Serialized machine learning model
├── train.csv # Training dataset
├── test.csv # Test dataset
├── requirements.txt # Project dependencies
├── README.md # Documentation
├── Images/ # EDA visual assets


---

## 📌 Dataset Details

- **Train dataset:** Includes features and target (Loan_Status)  
- **Test dataset:** Includes only input features  
- Number of rows: 614 in train and 367 in test  
- Number of columns: 13 in train and 12 in test :contentReference[oaicite:0]{index=0}

### Example Features:

| Feature            | Description                               |
|------------------|-------------------------------------------|
| Gender           | Male/Female                               |
| Education        | Graduate/Not Graduate                     |
| ApplicantIncome  | Income of applicant                       |
| LoanAmount       | Loan amount applied                       |
| Credit_History   | Whether credit history is met or not      |
| Property_Area    | Urban / Semiurban / Rural                 |

---

## 📊 Exploratory Data Analysis (EDA)

1. **Univariate Analysis**  
   - Visualized categorical and numerical features  
   - Bar plots for distributions  
   - Outlier detection and handling :contentReference[oaicite:1]{index=1}

2. **Bivariate Analysis**  
   - Plotted relationships between features and `Loan_Status`  
   - Found higher loan approval probability for applicants with credit history and higher income :contentReference[oaicite:2]{index=2}

3. **Correlation Analysis**  
   - Heatmap to find correlations  
   - Strong correlations observed between income and loan amount, and credit history and loan status :contentReference[oaicite:3]{index=3}

---

## 🛠 Data Preprocessing

- **Missing values:** Imputed using mode for categorical and median for numerical  
- **Outlier treatment:** Log transformation applied to skewed numerical variables  
- **Encoding:** Categorical variables encoded to numerical format  
- These steps ensure better model training and performance :contentReference[oaicite:4]{index=4}

---

## 🧩 Machine Learning Model

- **Model used:** Logistic Regression  
- **Validation technique:** Stratified Shuffle Split  
- **Performance:** Achieved ~84% accuracy and ~82% F1-score on validation data :contentReference[oaicite:5]{index=5}

---

## 🚀 Web Application (Streamlit)

The `app.py` script builds a simple UI where users can:

- Enter applicant details
- Click “Predict”
- Get loan approval prediction instantly

---

## 🌐 Live Demo (UI)

🚀 **Loan Prediction System – Streamlit App**  
👉 https://loan-prediction-system-klty6rb58c3sw7eiayswjb.streamlit.app/

Screenshots:
<img width="1499" height="873" alt="Screenshot 2026-01-06 142904" src="https://github.com/user-attachments/assets/22b9382d-9125-49fe-b6a6-85ee898294fb" />
<img width="772" height="509" alt="Screenshot 2026-01-06 142652" src="https://github.com/user-attachments/assets/10a5186a-86fb-4f71-a9b3-478d05e80137" />
