import time

import streamlit as st

from src.database_setup import Database_Setup
from src.llm_router import LLMRouter
from src.prompt_tier_classifier import PromptTierClassifier
from src.response_validator import ResponseValidator

# --- Page Configuration (ROOT LEVEL) ---
st.set_page_config(page_title="Nexus", page_icon="🌌", layout="wide")


# --- Shared Resource Initialization ---
@st.cache_resource
def load_nexus_engine():
    return {
        "memory": Database_Setup(),
        "tier_classifier": PromptTierClassifier(),
        "router": LLMRouter(),
        "validator": ResponseValidator(),
    }


if "engine" not in st.session_state:
    st.session_state.engine = load_nexus_engine()

if "history" not in st.session_state:
    st.session_state.history = []


# --- Logic Functions ---
def process_query(prompt, mode):
    if mode == "AUTO":
        st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        logs = []
        start_time = time.time()

        with st.status("Nexus Thinking...", expanded=False) as status:
            cached_res = None
            if mode == "AUTO":
                st.write("Checking L1 Cache...")
                cached_res = st.session_state.engine["memory"].check_cache(prompt)

            if cached_res:
                response, tier = cached_res, "CACHE"
                status.update(label="Memory Hit!", state="complete")
                st.markdown(response)
            else:
                st.write("Retrieving Context...")
                context = st.session_state.engine["memory"].get_context(prompt)

                if mode == "AUTO":
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

                stream = st.session_state.engine["router"].get_response_stream(
                    tier, prompt, context
                )
                response = st.write_stream(stream)
                status.update(label=f"Processed via {tier}", state="complete")

        latency = round(time.time() - start_time, 3)
        st.session_state.engine["memory"].save_interaction(
            prompt=prompt,
            response=response,
            tier=tier,
            latency=latency,
            logs=logs,
            model=st.session_state.engine["router"].get_model_name(tier),
        )

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": response,
                "user_prompt": prompt,
                "source": tier,
                "latency": latency,
            }
        )
        st.rerun()


# --- UI Rendering ---
st.title("Nexus")

for i, chat in enumerate(st.session_state.history):
    with st.chat_message(chat["role"]):
        if chat["role"] == "assistant":
            st.caption(f"{chat['source']} Model | {chat.get('latency', 0)}s")
            st.markdown(chat["content"])
            if i == len(st.session_state.history) - 1:
                with st.popover("Not satisfied? Retry with..."):
                    c1, c2, c3 = st.columns(3)
                    if c1.button("LITE", width="stretch"):
                        process_query(chat["user_prompt"], "LITE")
                    if c2.button("MID", width="stretch"):
                        process_query(chat["user_prompt"], "MID")
                    if c3.button("PRO", width="stretch"):
                        process_query(chat["user_prompt"], "PRO")
        else:
            st.markdown(chat["content"])

if prompt := st.chat_input("Query Nexus..."):
    process_query(prompt, "AUTO")
