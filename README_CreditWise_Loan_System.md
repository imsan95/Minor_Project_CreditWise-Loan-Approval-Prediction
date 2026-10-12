# CreditWise Loan System — Loan Approval Prediction

## Project Overview

**CreditWise Loan System** is a supervised machine learning classification project that predicts whether a loan application is likely to be **Approved (1)** or **Rejected (0)** using applicant, financial, employment, and credit-related information.

The project is framed around a fictional financial institution, SecureTrust Bank, which wants to support faster and more consistent initial loan application reviews. The model output is intended as decision support; it should not replace final human review.

## Project Note

> Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval.  
> Implemented Binary classification along with EDA, feature engineering & model evaluation (Precision, Recall, F1).

## Objectives

- Explore loan application data and understand the target distribution.
- Handle missing numerical and categorical values.
- Perform exploratory data analysis (EDA) using charts and summary statistics.
- Encode categorical features and scale numeric features.
- Train and compare Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.
- Evaluate models using Precision, Recall, F1-score, Accuracy, and a Confusion Matrix.
- Experiment with feature engineering using squared Credit Score and Debt-to-Income Ratio (DTI) features.

## Dataset

The project uses `loan_approval_data.csv`. Each row represents a loan applicant.

Key fields include:

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier; removed before model training |
| `Applicant_Income` | Applicant's monthly income |
| `Coapplicant_Income` | Co-applicant's monthly income |
| `Employment_Status` | Employment type |
| `Age` | Applicant age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration in months |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Urban, Semi-Urban, or Rural |
| `Education_Level` | Applicant's education level |
| `Gender` | Applicant's gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target: `1` = Approved, `0` = Rejected |

## Workflow

1. **Load and inspect data** — review rows, data types, summary statistics, and missing values.
2. **Handle missing values** — use the mean for numerical columns and most-frequent values for categorical columns.
3. **EDA** — inspect approval distribution, category counts, income distributions, box plots, and relationships between selected features and approval.
4. **Prepare features** — remove `Applicant_ID`, encode categorical variables, and separate the target from the input features.
5. **Split and scale** — use an 80/20 train-test split with `random_state=42`; fit `StandardScaler` on training data and transform the test data.
6. **Train and evaluate models**:
   - Logistic Regression
   - K-Nearest Neighbors (`n_neighbors=5`)
   - Gaussian Naive Bayes
7. **Feature engineering** — add `DTI_Ratio_sq` and `Credit_Score_sq`, then retrain and evaluate the models.

## Models and Evaluation

The notebook evaluates each model using:

- **Precision** — of the applications predicted as approved, how many were actually approved.
- **Recall** — of the actually approved applications, how many were correctly identified.
- **F1-score** — harmonic mean of precision and recall.
- **Accuracy** — proportion of correct predictions.
- **Confusion Matrix** — summarizes correct and incorrect classifications.

The notebook's written conclusion identifies **Naive Bayes as the best model based on Precision**. Check the notebook outputs to review the corresponding metric values.

## Tools and Libraries

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## Repository Structure

```text
CreditWise-Loan-System/
├── credit_wise(2).ipynb
├── loan_approval_data.csv
├── CreditWise Loan System.pdf
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install Python and Jupyter Notebook.
3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. Keep `credit_wise(2).ipynb` and `loan_approval_data.csv` in the same folder.
5. Open Jupyter Notebook and run the notebook cells in order.

## Important Considerations

- This is a learning/minor project and should not be used to make real loan decisions without rigorous validation, monitoring, explainability, and human oversight.
- Loan decisions can affect people significantly. Precision and recall should be interpreted alongside the costs of false approvals and false rejections.
- For a production-ready pipeline, fit imputers and encoders only on training data (for example, by using a Scikit-learn `Pipeline` and `ColumnTransformer`) to avoid data leakage. Evaluate fairness across relevant groups before deployment.

## Author

**Santosh Dash**
