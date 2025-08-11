import streamlit as st
import pandas as pd
import pickle
from chatbot_utils import (
    unified_scheme_query,
    retriever,
    feature_cols,
    target_cols,
    scheme_df,
    model,
    qa_chain
)


st.set_page_config(layout="wide")
st.title(" Smart Scheme Assistant")

tab1, tab2 = st.tabs([" Scheme Recommendation", "💬 Ask our Assistant"])

# --------------------------
#  Tab 1: Scheme Recommender
# --------------------------
with tab1:
    st.header("Get Recommended Government Schemes")

    with st.form("recommendation_form"):
        st.markdown("###  Your Profile")

        user_input = {}

        # # ----- Numeric fields -----
        # col1, col2 = st.columns(2)
        # with col1:
        #     user_input["min_age"] = st.number_input("Minimum Age", min_value=0, max_value=150, step=1)
        #     user_input["annual_income"] = st.number_input("Annual Income (₹)", min_value=0)
        # with col2:
        #     user_input["max_age"] = st.number_input("Maximum Age", min_value=0, max_value=150, step=1)
        #     user_input["max_income"] = st.number_input("Maximum Eligible Income (if known)", min_value=0)

        # ------ Grouped feature checkboxes ------
        with st.expander(" Education & Student Status"):
            user_input["is_student"] = int(st.checkbox("Are you a student?"))
            user_input["is_school_student"] = int(st.checkbox("Are you a school student?"))
            user_input["is_college_student"] = int(st.checkbox("Are you a college student?"))
            user_input["has_graduated"] = int(st.checkbox("Have you graduated?"))
            user_input["is_dropout"] = int(st.checkbox("Are you a dropout?"))
            user_input["is_first_generation_learner"] = int(st.checkbox("First-generation learner?"))

        with st.expander(" Gender & Family"):
            user_input["is_female"] = int(st.checkbox("Are you a woman?"))
            user_input["is_male"] = int(st.checkbox("Are you a man?"))
            user_input["is_transgender"] = int(st.checkbox("Are you a transgender person?"))
            user_input["is_girl_child"] = int(st.checkbox("Are you a girl child?"))
            user_input["is_single_woman"] = int(st.checkbox("Single woman?"))
            user_input["has_large_family"] = int(st.checkbox("Large family?"))
            user_input["is_woman_headed_household"] = int(st.checkbox("Woman-headed household?"))
            user_input["is_maternal_beneficiary"] = int(st.checkbox("Pregnant/lactating mother?"))
            user_input["is_widow"] = int(st.checkbox("Are you a widow?"))
            user_input["is_child_of_single_parent"] = int(st.checkbox("Child of a single parent?"))

        with st.expander(" Residence & Housing"):
            user_input["is_rural_resident"] = int(st.checkbox("Do you live in a rural area?"))
            user_input["is_urban_resident"] = int(st.checkbox("Do you live in an urban area?"))
            user_input["is_migrant_worker"] = int(st.checkbox("Are you a migrant worker?"))
            user_input["lives_in_slum"] = int(st.checkbox("Live in a slum area?"))
            user_input["is_homeless"] = int(st.checkbox("Are you homeless?"))
            user_input["has_no_house"] = int(st.checkbox("Do you have no house?"))
            user_input["has_pukka_house"] = int(st.checkbox("Do you live in a pucca house?"))
            user_input["has_kutcha_house"] = int(st.checkbox("Do you live in a kutcha house?"))

        with st.expander(" Employment & Economic"):
            user_input["is_farmer"] = int(st.checkbox("Are you a farmer?"))
            user_input["is_self_employed"] = int(st.checkbox("Self-employed?"))
            user_input["is_unemployed"] = int(st.checkbox("Unemployed?"))
            user_input["is_salaried_employee"] = int(st.checkbox("Salaried employee?"))
            user_input["is_daily_wage_worker"] = int(st.checkbox("Daily wage worker?"))
            user_input["is_job_seeker"] = int(st.checkbox("Actively looking for a job?"))
            user_input["is_worker"] = int(st.checkbox("Are you a worker/labourer?"))
            user_input["is_labour"] = int(st.checkbox("Labour-intensive job?"))

        with st.expander(" Health, Disability & Special Status"):
            user_input["is_disabled"] = int(st.checkbox("Do you have a disability?"))
            user_input["has_chronic_illness"] = int(st.checkbox("Chronic illness?"))
            user_input["is_senior_citizen"] = int(st.checkbox("Senior citizen?"))
            user_input["is_orphan"] = int(st.checkbox("Are you an orphan?"))
            user_input["is_person_with_hiv"] = int(st.checkbox("Living with HIV?"))
            user_input["is_victim_of_abuse"] = int(st.checkbox("Victim of abuse or violence?"))
            user_input["has_disabled_family_member"] = int(st.checkbox("Have a disabled family member?"))

        with st.expander(" Caste, Minority, Identity"):
            user_input["is_sc_st"] = int(st.checkbox("Do you belong to SC/ST?"))
            user_input["is_obc"] = int(st.checkbox("Do you belong to OBC?"))
            user_input["belongs_to_minority"] = int(st.checkbox("Belong to a religious/language minority?"))
            user_input["is_tribal"] = int(st.checkbox("Do you belong to a tribal community?"))

        with st.expander(" Financial & Govt. Services"):
            user_input["has_bank_account"] = int(st.checkbox("Have a bank account?"))
            user_input["no_asset_ownership"] = int(st.checkbox("No asset ownership?"))
            user_input["receives_pension"] = int(st.checkbox("Already receiving a pension?"))
            user_input["receives_existing_govt_benefits"] = int(st.checkbox("Already receiving govt benefits?"))
            user_input["has_family_member_in_govt_service"] = int(st.checkbox("Family member in govt service?"))

        with st.expander(" Other Community Background"):
            user_input["belongs_to_fisherfolk"] = int(st.checkbox("From a fisherfolk community?"))
            user_input["belongs_to_weaver_community"] = int(st.checkbox("From a weaver community?"))
            user_input["is_ex_serviceman"] = int(st.checkbox("Are you an ex-serviceman?"))

        # ----- Numeric fields -----
        col1, col2 = st.columns(2)
        with col1:
            user_age = st.number_input("Your Age", min_value=0, max_value=150, step=1)
            user_input["min_age"] = user_age
            user_input["max_age"] = user_age
            user_input["annual_income"] = st.number_input("Annual Income (₹)", min_value=0)
        with col2:
            user_input["max_income"] = st.number_input("Maximum Eligible Income (if known)", min_value=0)
        st.subheader("Preferences:")
        gov_pref = st.selectbox("Preferred Government Type", ["all", "central", "state"])
        available_states = sorted(scheme_df["State"].dropna().unique())
        state_name = st.selectbox("Select your State (optional)", [""] + available_states)
        query = st.text_input("Optional Query (e.g., scholarships for disabled)")

        

        num_results = st.slider("Number of schemes to display", min_value=5, max_value=50, value=10)

        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        results, method = unified_scheme_query(
            query=query or "scheme recommendation",
            user_input=user_input,
            model=model,
            feature_cols=feature_cols,
            target_cols=target_cols,
            scheme_df=scheme_df,
            retriever=retriever,
            gov_pref=gov_pref,
            state_name=state_name.strip() or None,
            top_k=num_results
        )
    
        st.markdown(f"**Results via:** `{method}`")
    
        selected_features = [k.replace("_", " ") for k, v in user_input.items() if v not in [0, "", None]]
        if selected_features:
            st.success(f"Recommended based on: {', '.join(selected_features)}")
    
        if not results.empty:
            # Make Scheme_Name clickable
            results["Scheme_Name"] = results.apply(
                lambda row: f'<a href="{row["URL"]}" target="_blank">{row["Scheme_Name"]}</a>' if pd.notna(row["URL"]) else row["Scheme_Name"],
                axis=1
            )
    
            # Choose which columns to show (exclude original URL column if needed)
            display_cols = ["Scheme_Name", "State", "Ministry"]
            if "match_score" in results.columns:
                display_cols.append("match_score")
    
            # Convert to HTML table for clickable links
            styled_df = results[display_cols].to_html(escape=False, index=False)
    
            # Display as styled table
            st.markdown("### Recommended Schemes:")
            st.markdown(styled_df, unsafe_allow_html=True)
    
        else:
            st.warning("No matching schemes found.")

# --------------------------
#  Tab 2: Open-ended Chatbot
# --------------------------
with tab2:
    st.header("Chat with SchemeBot ")
    user_query = st.text_input("Ask something about the Indian Government Schemes...")

    if st.button("Ask"):
        if user_query.strip():
            try:
                response = qa_chain.run(user_query)
                st.markdown("**Answer:**")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a question.")

