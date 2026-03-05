# Nexus: A Smarter Way to Route AI Queries

Nexus is an intelligent orchestration layer built to stop wasting high-compute resources on low-complexity tasks. By analyzing every prompt before it hits an LLM, Nexus routes queries to the most efficient model tier—saving cost, reducing latency, and cutting down on the environmental footprint of your AI operations.

## 🌌 What’s Inside

- **Smart Tier Routing**: Instead of one-size-fits-all, Nexus uses a mix of ML classification and heuristics to pick between **LITE**, **MID**, or **PRO** tiers.
- **Two-Layer Memory**:
  - **L1 Cache**: Recognizes identical questions instantly using ChromaDB for zero-latency responses.
  - **L2 Context**: Pulls in relevant past interactions so the AI actually remembers your history.
- **Self-Healing Logic**: If a high-end model fails or hits a limit, the system automatically falls back to a secondary model to keep the conversation going.
- **Sustainability Tracking**: Every time you route to a lighter model, Nexus calculates the CO2 and water saved compared to using a heavy model.
- **Deep Observability**: A built-in dashboard tracks model distribution, latency, and "intelligence traces" to show exactly why a tier was chosen.

## 🛠️ The Tech Stack

Nexus is built with Python and Streamlit, keeping the logic modular and easy to tweak:

- **The Brain (`Home.py`)**: The central hub that manages the chat flow and coordinates between the classifier and the router.
- **The Router (`llm_router.py`)**: Handles the actual handshakes with Gemini models via LiteLLM.
- **The Classifier (`prompt_tier_classifier.py`)**: Uses semantic embeddings to judge if a prompt is "simple" or "complex".
- **The Storage (`database_setup.py`)**: A dual-threat setup using SQLite for logs and ChromaDB for vector-based memory.
- **The Guardrail (`response_validator.py`)**: A multi-stage validator that checks for code quality and ensures the AI didn't just give a generic refusal.

## 🚀 Getting Started

### 1. Requirements

- Python 3.10+
- A Gemini API Key

### 2. Quick Setup

```bash
# Get the code
git clone <your-repo-url>
cd nexus

# Install the essentials
pip install -r requirements.txt

```

### 3. Configure Your Environment

Create a `.env` file in the root directory:

```text
GEMINI_API_KEY=your_key_here

```

### 4. Launch

```bash
streamlit run Home.py

```

## 📊 Monitoring & Safety

You can switch to the **Dashboard** page to see your efficiency metrics in real-time. Nexus also runs a **Response Validator** on every MID/PRO response to ensure the output is actually helpful and technically sound before you ever see it.
