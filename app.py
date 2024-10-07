import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
df = pd.read_csv("credit_train.csv", na_values=['x'])
df.drop(columns=['Customer_ID', 'ID', 'Unnamed: 0'], inplace=True)

# Load the saved KNN pipeline
with open('random_forest_model1.pkl', 'rb') as file:
    pipeline_knn = joblib.load(file)

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Personal Information", "Visualizations", "Credit Score Prediction"])

# Page 1: Personal Information
if page == "Personal Information":
    st.title("Personal Information")
    st.write("Name: Yusuf")
    st.write("Age: 22")
    st.write("Email: yadel9989@gmail.com")
    st.write("University: MTI University")
    st.write("Contact: +01140952160")

    # Brief Description
    st.subheader("Brief Description")
    st.write("""
        I am a data scientist with expertise in machine learning, data analysis, and model deployment.
        My projects include real estate price prediction and credit score classification.
    """)

    # Image (add your image path)
    st.subheader("My Image")
    st.image("joe.jpg", caption="Joe Adel", use_column_width=True)

# Page 2: Visualizations
elif page == "Visualizations":
    st.title("Data Visualizations")

    # Drop any specific outliers if needed
    df = df.drop(54739, axis=0)

    # Categorical column visualizations
    categorical_cols = ['Month', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Pie chart for Occupation
    st.subheader("Occupation Distribution")
    plt.figure(figsize=(10, 5))
    df["Occupation"].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Occupation Distribution')
    st.pyplot(plt)

    # Numerical columns for histograms
    numerical_cols = ['Monthly_Inhand_Salary', 'Delay_from_due_date', 'Num_Credit_Inquiries',
                      'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                      'Amount_invested_monthly', 'Monthly_Balance', 'Credit_Score', 'Num_Bank_Accounts']

    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        st.pyplot(plt)

    # Bar chart for Credit Mix
    st.subheader("Credit Mix Distribution")
    plt.figure(figsize=(10, 5))
    df["Credit_Mix"].value_counts().plot.bar()
    plt.title('Credit Mix Distribution')
    st.pyplot(plt)

    # Boxplot: Credit Score vs Monthly Inhand Salary
    st.subheader("Credit Score vs Monthly Inhand Salary")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Credit_Score', y='Monthly_Inhand_Salary', data=df)
    st.pyplot(plt)

    # Bar plot: Monthly Inhand Salary vs Outstanding Debt (Top 10)
    st.subheader("Monthly Inhand Salary vs Outstanding Debt (Top 10)")
    plt.figure(figsize=(10, 5))
    df[["Monthly_Inhand_Salary", "Outstanding_Debt"]].head(10).plot.bar()
    plt.title('Monthly Inhand Salary vs Outstanding Debt')
    st.pyplot(plt)

    # Barplot: Credit Score vs Credit Mix
    st.subheader("Credit Score vs Credit Mix")
    grouped_data = df.groupby('Credit_Mix')['Credit_Score'].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=grouped_data.index, y=grouped_data)
    plt.xlabel('Credit Mix')
    plt.ylabel('Credit Score')
    plt.title('Credit Score vs Credit Mix')
    st.pyplot(plt)

    # Heatmap of correlations
    st.subheader("Correlation Heatmap")
    heatmap = df.select_dtypes(exclude='object').corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap, annot=True)
    st.pyplot(plt)

# Page 3: Credit Score Prediction
elif page == "Credit Score Prediction":
    st.title("Credit Score Prediction")

    # Input fields
    annual_income = st.number_input('Annual Income', min_value=0.0, value=0.0)
    monthly_salary = st.number_input('Monthly Inhand Salary', min_value=0.0, value=0.0)
    num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, value=0)
    num_credit_cards = st.number_input('Number of Credit Cards', min_value=0, value=0)
    interest_rate = st.number_input('Interest Rate', min_value=0, value=0)
    num_of_loans = st.number_input('Number of Loans', min_value=0, value=0)
    changed_credit_limit = st.number_input('Changed Credit Limit', min_value=0, value=0)
    num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, value=0)
    credit_mix = st.number_input('Credit Mix', min_value=0, value=0)
    outstanding_debt = st.number_input('Outstanding Debt', min_value=0.0, value=0.0)
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, max_value=1.0, value=0.0)

    # Button to make prediction
    if st.button('Predict Credit Score'):
        # Prepare input data
        input_data = pd.DataFrame({
            'Annual_Income': [annual_income],
            'Monthly_Inhand_Salary': [monthly_salary],
            'Num_Bank_Accounts': [num_bank_accounts],
            'Num_Credit_Card': [num_credit_cards],
            'Interest_Rate': [interest_rate],
            'Num_of_Loan': [num_of_loans],
            'Changed_Credit_Limit': [changed_credit_limit],
            'Num_Credit_Inquiries': [num_credit_inquiries],
            'Credit_Mix': [credit_mix],
            'Outstanding_Debt': [outstanding_debt],
            'Credit_Utilization_Ratio': [credit_utilization_ratio],
        })

        # Predict the credit score
        prediction = pipeline_knn.predict(input_data)

        # Show the prediction
        st.write(f'Predicted Credit Score: {prediction[0]}')
