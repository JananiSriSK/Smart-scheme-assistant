import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import joblib

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=google_api_key,
    convert_system_message_to_human=True
)

# Initialize Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# Utility Functions
def create_feature_vector(user_input_dict, feature_cols):
    X_user = pd.DataFrame([0]*len(feature_cols), index=feature_cols).T
    for key, value in user_input_dict.items():
        if key in feature_cols:
            X_user.loc[0, key] = value
    return X_user

def filter_by_government_level(df, preference="all", state_name=None):
    if preference.lower() == "central":
        return df[df["Scheme_Level"] == "Central"]
    elif preference.lower() == "state":
        if state_name:
            return df[
                (df["Scheme_Level"] == "State") &
                (df["State"].str.lower() == state_name.lower())
            ]
        else:
            return df[df["Scheme_Level"] == "State"]
    elif preference.lower() == "all":
        if state_name:
            return df[
                (df["Scheme_Level"] == "Central") |
                ((df["Scheme_Level"] == "State") & (df["State"].str.lower() == state_name.lower()))
            ]
        else:
            return df
    else:
        return df

def build_document(row):
    content = f"""
    Scheme Name: {row['Scheme_Name']}
    State: {row['State']}
    Ministry: {row['Ministry']}
    Scheme Level: {row['Scheme_Level']}
    Eligibility: {row['Eligibility']}
    Benefits: {row['Benefits']}
    Details: {row['Details']}
    Application Process: {row['Application_Process']}
    Documents Required: {row['Documents_Required']}
    URL: {row['URL']}
    """.strip()
    metadata = {
        "scheme_name": row["Scheme_Name"],
        "state": row["State"],
        "ministry": row["Ministry"],
        "scheme_level": row["Scheme_Level"],
        "url": row["URL"],
    }
    return Document(page_content=content, metadata=metadata)

def convert_documents_to_dataframe(documents):
    data = []
    for doc in documents:
        meta = doc.metadata
        row = {
            "Scheme_Name": meta.get("scheme_name", ""),
            "State": meta.get("state", ""),
            "Ministry": meta.get("ministry", ""),
            "URL": meta.get("url", ""),
            "Content": doc.page_content[:500] + "..."
        }
        data.append(row)
    return pd.DataFrame(data)

def build_answer_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return chain

class GeminiRetriever:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever()

    def invoke(self, query, top_k=5):
        docs = self.retriever.invoke(query)
        return pd.DataFrame([doc.metadata for doc in docs[:top_k]])

def unified_scheme_query(
    query,
    user_input,
    model,
    feature_cols,
    target_cols,
    scheme_df,
    retriever,
    gov_pref="all",
    state_name=None,
    top_k=5,
    min_ml_results=3,
    min_match_ratio=0.5  # at least 50% of predicted labels must match
):
    # Step 1: Prepare input for ML model
    X_input = np.array([[user_input.get(col, 0) for col in feature_cols]])
    predicted = model.predict(X_input)
    predicted_labels = [target_cols[i] for i, val in enumerate(predicted[0]) if val == 1]

    # Step 2: Filter scheme_df by government level and state
    filtered_df = filter_by_government_level(scheme_df, gov_pref, state_name)

    # Step 3: Get valid predicted labels present in filtered_df
    valid_labels = [label for label in predicted_labels if label in filtered_df.columns]

    # Step 4: Match based on partial label overlap
    if valid_labels:
        label_matrix = filtered_df[valid_labels].apply(pd.to_numeric, errors="coerce").fillna(0)
        label_sum = label_matrix.sum(axis=1)
        min_required = max(1, int(len(valid_labels) * min_match_ratio))

        filtered_df["match_score"] = label_sum / len(valid_labels)
        ml_results = filtered_df[label_sum >= min_required]
        ml_results = ml_results.sort_values(by="match_score", ascending=False)
    else:
        ml_results = pd.DataFrame()

    # Step 5: Return ML results if sufficient and state-matching
    if (
        not ml_results.empty and
        len(ml_results) >= min_ml_results and
        (state_name is None or any(ml_results["State"] == state_name))
    ):
        return ml_results.head(top_k), "ML"

    # Step 6: Fallback to Retriever (RAG)
    try:
        rag_docs = retriever.invoke(query)
        rag_results = convert_documents_to_dataframe(rag_docs)

        if state_name:
            rag_results = rag_results[rag_results["State"].isin([state_name, "All India"])]

        return rag_results.head(top_k), "Retriever"

    except Exception as e:
        print("Retriever failed:", e)
        return pd.DataFrame(), "Retriever"


# Load FAISS vectorstore
vectorstore = FAISS.load_local("scheme_vector_index_gemini", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Load saved model and metadata
model = joblib.load("trained_model_xgb.pkl")
feature_cols = joblib.load("feature_columns.pkl")
target_cols = joblib.load("target_columns.pkl")
scheme_df = pd.read_csv("./datasets/final_gemini_extract.csv")

# Classify scheme level
scheme_df["Ministry"] = scheme_df["Ministry"].fillna("")
def classify_scheme(row):
    if row["State"] == "All India" or row["Ministry"].strip() != "":
        return "Central"
    else:
        return "State"
scheme_df["Scheme_Level"] = scheme_df.apply(classify_scheme, axis=1)

# Build QA chain
qa_chain = build_answer_chain(vectorstore)
