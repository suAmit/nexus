import os
import sqlite3

import pandas as pd
import streamlit as st


def render():

    st.set_page_config(
        page_title="Nexus | Database",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.header("Database Explorer")

    tab1, tab2 = st.tabs(["Relational Logs (SQL)", "Vector Memory (Chroma)"])

    with tab1:
        st.subheader("Interaction History")
        conn = sqlite3.connect("data/nexus_logs.db")

        df = pd.read_sql_query(
            "SELECT id, timestamp, tier, model_version, prompt, response FROM interaction_logs ORDER BY id DESC",
            conn,
        )
        conn.close()

        # Search filter - Updated to only search prompt/response since process_logs is gone
        search = st.text_input("Search prompts or responses...")
        if search:
            df = df[
                df["prompt"].str.contains(search, case=False)
                | df["response"].str.contains(search, case=False)
            ]

        st.dataframe(df, width="stretch", hide_index=True)

        if st.button("Clear SQL Logs", type="secondary"):
            conn = sqlite3.connect("data/nexus_logs.db")
            conn.execute("DELETE FROM interaction_logs")
            conn.commit()
            conn.close()
            st.success("Logs cleared.")
            st.rerun()

    with tab2:
        st.subheader("Semantic Store Management (Collapsed JSON)")

        try:
            mem_engine = st.session_state.engine["memory"]
            collection = mem_engine.memory_collection

            raw_data = collection.get()
            count = len(raw_data["ids"])

            st.metric("Total Memories", count)

            if count > 0:
                for i in reversed(range(count)):
                    with st.container(border=True):
                        st.markdown(f"**Prompt:** `{raw_data['documents'][i]}`")

                        entry_json = {
                            "id": raw_data["ids"][i],
                            "document": raw_data["documents"][i],
                            "metadata": raw_data["metadatas"][i],
                        }

                        st.json(entry_json, expanded=False)

                st.divider()
                if st.button("Wipe Vector Memory", type="primary"):
                    mem_engine.chroma_client.delete_collection("semantic_memory")
                    mem_engine._init_chroma()
                    st.rerun()
            else:
                st.info("No semantic data found.")

        except Exception as e:
            st.error(f"Error rendering: {e}")


render()
