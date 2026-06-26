import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

st.set_page_config(page_title="Marketing Campaign Prediction", layout="wide")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_excel("marketing_campaign.xlsx")

df = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Admin Page",
        "Train Models",
        "User Input",
        "Chatbot",
        "Logout"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("Marketing Campaign Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Responses", int(df["Response"].sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Response Distribution")
    st.bar_chart(df["Response"].value_counts())

# ---------------- ADMIN PAGE ----------------

elif page == "Admin Page":

    st.title("Admin Page")

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistics")
    st.write(df.describe())

# ---------------- TRAIN MODELS ----------------

elif page == "Train Models":

    st.title("Train Machine Learning Models")

    if st.button("Train All Models"):

        data = df.copy()

        data = data.fillna(0)

        data["Dt_Customer"] = pd.to_datetime(
            data["Dt_Customer"],
            errors="coerce"
        )

        data["Dt_Customer"] = (
            data["Dt_Customer"]
            .astype("int64") // 10**9
        )

        le = LabelEncoder()

        for col in ["Education", "Marital_Status"]:
            data[col] = le.fit_transform(
                data[col].astype(str)
            )

        X = data.drop("Response", axis=1)
        y = data["Response"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_acc = accuracy_score(
            y_test,
            dt.predict(X_test)
        )

        # Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(
            y_test,
            rf.predict(X_test)
        )

        # SVM
        svm = SVC()
        svm.fit(X_train_scaled, y_train)
        svm_acc = accuracy_score(
            y_test,
            svm.predict(X_test_scaled)
        )

        # XGBoost
        xgb = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

        xgb.fit(X_train, y_train)

        xgb_acc = accuracy_score(
            y_test,
            xgb.predict(X_test)
        )

        result = pd.DataFrame({
            "Model": [
                "Decision Tree",
                "Random Forest",
                "SVM",
                "XGBoost"
            ],
            "Accuracy": [
                dt_acc,
                rf_acc,
                svm_acc,
                xgb_acc
            ]
        })

        st.subheader("Accuracy Comparison")
        st.dataframe(result)

        st.bar_chart(
            result.set_index("Model")
        )

# ---------------- USER INPUT ----------------

elif page == "User Input":

    st.title("Customer Response Prediction")

    income = st.number_input("Income", value=50000)

    recency = st.number_input("Recency", value=10)

    wines = st.number_input("MntWines", value=100)

    web = st.number_input("NumWebPurchases", value=5)

    catalog = st.number_input(
        "NumCatalogPurchases",
        value=2
    )

    if st.button("Predict"):

        score = (
            income
            + wines * 20
            + web * 100
            + catalog * 100
        )

        if score > 60000:
            st.success(
                "Likely to Respond to Campaign"
            )
        else:
            st.error(
                "Not Likely to Respond"
            )

# ---------------- CHATBOT ----------------

elif page == "Chatbot":

    st.title("Marketing Chatbot")

    question = st.text_input(
        "Ask a question"
    )

    if st.button("Send"):

        q = question.lower()

        if "response" in q:
            st.write(
                "Response indicates campaign acceptance."
            )

        elif "income" in q:
            st.write(
                "Income is one of the strongest predictors."
            )

        elif "best model" in q:
            st.write(
                "Check Train Models page for best accuracy."
            )

        else:
            st.write(
                "Please ask about the marketing dataset."
            )

# ---------------- LOGOUT ----------------

elif page == "Logout":

    st.title("Logout")

    st.success(
        "Logged Out Successfully"
    )