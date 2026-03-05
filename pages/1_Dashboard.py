import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


def render():
    st.set_page_config(
        page_title="Nexus | Dashboard",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.header("System Dashboard")

    try:
        conn = sqlite3.connect("data/nexus_logs.db")
        logs_df = pd.read_sql_query("SELECT * FROM interaction_logs", conn)
        conn.close()

        if logs_df.empty:
            st.info("No interaction data found. Generate some logs in the Chat first!")
            return

        # --- 1. KEY PERFORMANCE INDICATORS (KPIs) ---
        # actual_cost = df["cost"].sum()
        # Savings calculation: Assuming a base cost for 'PRO' vs actual used
        # est_savings = (len(df) * 0.005) - actual_cost

        # c1, c2, c3, c4 = st.columns(4)
        # c1.metric("Total Requests", len(df))
        # c2.metric("Avg Latency", f"{df['latency'].mean():.2f}s")
        # c3.metric("Cache/L2 Hits", len(df[df["tier"].isin(["CACHE", "L2"])]))
        # c4.metric("Est. Savings", f"${max(0, est_savings):.4f}")

        # st.divider()

        # --- 2. DISTRIBUTION CHARTS ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Tier Allocation")
            color_map = {
                "LITE": "#00CC96",
                "MID": "#636EFA",
                "PRO": "#EF553B",
                "CACHE": "#FECB52",
            }
            fig_tier = px.pie(
                logs_df,
                names="tier",
                hole=0.6,
                color="tier",
                color_discrete_map=color_map,
            )
            fig_tier.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_tier, width="stretch")

        with col_right:
            st.subheader("Model Usage")

            # 1. Filter out nulls, NaNs, and "unknown" strings
            # We use .copy() to avoid SettingWithCopy warnings
            filtered_df = logs_df[
                logs_df["model_version"].notna()
                & (logs_df["model_version"] != "")
                & (logs_df["model_version"].str.lower() != "unknown")
            ].copy()

            # 2. Proceed with counting using the filtered data
            model_counts = filtered_df["model_version"].value_counts().reset_index()
            model_counts.columns = ["Model", "Count"]

            # 3. Render Chart
            fig_model = px.bar(
                model_counts,
                x="Count",
                y="Model",
                orientation="h",
                color="Count",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_model, width="stretch")
        # with col_right:
        #     st.subheader("Model Usage")
        #     model_counts = logs_df["model_version"].value_counts().reset_index()
        #     model_counts.columns = ["Model", "Count"]
        #     fig_model = px.bar(
        #         model_counts,
        #         x="Count",
        #         y="Model",
        #         orientation="h",
        #         color="Count",
        #         color_continuous_scale="Viridis",
        #     )
        #     st.plotly_chart(fig_model, width="stretch")

        # --- 3. RECENT TRACES ---
        st.divider()
        st.subheader("Recent Intelligence Traces")
        display_df = logs_df[["timestamp", "tier", "model_version", "latency"]].tail(4)
        # display_df = df[["timestamp", "tier", "model_version", "latency", "cost"]].tail(
        #     5
        # )
        st.table(display_df.sort_values(by="timestamp", ascending=False))

        # --- 4. SHAREABLE IMPACT CARD (THE NEW SECTION) ---
        # st.divider()

        # SUSTAINABILITY CALCULATIONS
        # Logic: 1 LITE query saves ~0.27g CO2 and ~0.5mL water vs a heavy PRO query.
        # lite_calls = len(df[df["tier"].isin(["LITE", "CACHE"])])
        # total_calls = len(df)
        # co2_val = lite_calls * 0.27
        # water_val = lite_calls * 0.5
        # efficiency_pct = (lite_calls / total_calls * 100) if total_calls > 0 else 0
        #
        # st.markdown("### 🌍 Your Nexus Sustainability Impact")
        #
        # with st.container(border=True):
        #     s1, s2, s3 = st.columns(3)
        #
        #     with s1:
        #         st.metric("Total CO₂ Saved", f"{co2_val:.2f}g", "🌱 Clean AI")
        #
        #     with s2:
        #         st.metric("Water Saved", f"{water_val:.1f}mL", "💧 Cooling Efficiency")
        #
        #     with s3:
        #         st.metric("Energy Efficiency", f"{efficiency_pct:.1f}%", "⚡ Optimized")
        #
        #     st.write(
        #         f"**By routing {lite_calls} requests to efficiency tiers, you've reduced your AI footprint!**"
        #     )
        #
        #     # Simple text area for easy copy-pasting to social media
        #     share_text = f"My Nexus AI system saved {co2_val:.2f}g of CO2 and {water_val:.1f}mL of water today using intelligent routing! 🌱💧 #GreenAI #Sustainability"
        #     st.text_area("Share your impact:", share_text, height=70)
        #
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


render()
