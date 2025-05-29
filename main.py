import streamlit as st
import pandas as pd
from orchestrator import run_flow

st.title("Agentic AI Banking Compliance Demo")

uploaded_file = st.file_uploader("Upload banking data (.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    if st.button("Run Agentic Compliance Flow"):
        with st.spinner("Running agentic flow..."):
            state, _ = run_flow(df)

        st.success("Agentic flow completed!")
        st.markdown(f"**Result:** {state.result}")
        if state.error:
            st.error(f"Error: {state.error}")

        # Show non-compliant accounts table if present
        non_compliant = getattr(state, 'non_compliant_data', None)
        if non_compliant is not None and not non_compliant.empty:
            st.write("### Non-Compliant Accounts")
            st.dataframe(non_compliant)

            # Download button for non-compliant accounts
            csv = non_compliant.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Non-Compliant Accounts as CSV",
                data=csv,
                file_name='non_compliant_accounts.csv',
                mime='text/csv'
            )
        else:
            st.info("No non-compliant accounts found in this run.")