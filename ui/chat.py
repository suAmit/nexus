import time

import streamlit as st


def render():

    st.set_page_config(
        page_title="Nexus | Chat",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Nexus Orchestrator")

    if "history" not in st.session_state:
        st.session_state.history = []

    # --- 1. DISPLAY HISTORY ---
    for i, chat in enumerate(st.session_state.history):
        with st.chat_message(chat["role"]):
            if chat["role"] == "assistant":
                st.caption(f"{chat['source']} Model")
                st.markdown(chat["content"])

                # RE-TRY LOGIC: Only show for the very last message
                if i == len(st.session_state.history) - 1:
                    with st.popover("Not satisfied? Retry with..."):
                        retry_col1, retry_col2, retry_col3 = st.columns(3)
                        if retry_col1.button("LITE", width="stretch"):
                            process_query(chat["user_prompt"], "LITE")
                        if retry_col2.button("MID", width="stretch"):
                            process_query(chat["user_prompt"], "MID")
                        if retry_col3.button("PRO", width="stretch"):
                            process_query(chat["user_prompt"], "PRO")
            else:
                st.markdown(chat["content"])

    # --- 2. THE CLEAN INPUT ---
    if prompt := st.chat_input("Query Nexus..."):
        process_query(prompt, "AUTO")


# --- 3. THE CORE PROCESSING FUNCTION ---
def process_query(prompt, mode):
    # Immediate User Display
    if mode == "AUTO":  # Only add to history if it's a new prompt
        st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        logs = []
        start_time = time.time()

        with st.status("Nexus Thinking...", expanded=False) as status:
            # L1: Cache (Skip if manual retry)
            cached_res = None
            if mode == "AUTO":
                st.write("Checking L1 Cache...")
                cached_res = st.session_state.engine["memory"].check_cache(prompt)

            if cached_res:
                response, tier = cached_res, "CACHE"
                status.update(label="Memory Hit!", state="complete")
                st.markdown(response)
            else:
                # L2: Context
                st.write("Retrieving Context...")
                context = st.session_state.engine["memory"].get_context(prompt)

                # L3: Scoring (Auto vs Manual)
                if mode == "AUTO":
                    # Flexible unpacking: captures whatever comes back
                    scorer_output = st.session_state.engine[
                        "tier_classifier"
                    ].predict_tier(prompt)

                    if len(scorer_output) == 3:
                        tier, conf, detail = scorer_output
                        status_msg = f"Auto-selected: {tier} ({detail})"
                    else:
                        tier, conf = scorer_output
                        status_msg = f"Auto-selected: {tier} ({conf:.1%})"

                    status.update(label=status_msg)
                else:
                    tier = mode
                    status.update(label=f"Manual Override: {tier}")
                # L3: Scoring (Auto vs Manual)
                # if mode == "AUTO":
                #     tier, conf, detail = st.session_state.engine["scr"].predict_tier(
                #         prompt
                #     )
                #     status.update(label=f"Auto-selected: {tier} ({detail})")
                # else:
                #     tier = mode
                #     status.update(label=f"Manual Override: {tier}")

                # L4: Streaming
                stream = st.session_state.engine["router"].get_response_stream(
                    tier, prompt, context
                )
                response = st.write_stream(stream)

                status.update(label=f"Processed via {tier}", state="complete")

        # Persistence
        latency = round(time.time() - start_time, 3)
        st.session_state.engine["memory"].save_interaction(
            prompt=prompt,
            response=response,
            tier=tier,
            latency=latency,
            logs=logs,
            model=st.session_state.engine["router"].get_model_name(tier),
        )

        # Update History
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": response,
                "user_prompt": prompt,  # Store original prompt to allow retries
                "source": tier,
                "latency": latency,
            }
        )
        st.rerun()
