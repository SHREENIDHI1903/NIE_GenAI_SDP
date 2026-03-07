# Deterministic Workflow Builder 🛠️

Welcome to the **Deterministic Workflow Builder** project! This repository is designed to teach students how to bridge the gap between human language and machine-readable data structures using Large Language Models (LLMs). 

In traditional software, we expect data in strict formats (like JSON) to pass through our APIs and databases. However, generative AI models natively output unstructured, unpredictable text. This project introduces the concept of **Deterministic Workflows**—engineering systems that force LLMs to output reliable, structured data every single time.

## 🎯 Learning Objectives

By exploring this project, you will learn:
1.  **Meta-Prompting**: How to use an LLM (the "Prompt Engineer") to write optimized system prompts for *another* LLM (the "Worker").
2.  **Structural Output Parsing**: How to force local models like Gemma to return data matching a strict schema rather than polite conversational replies.
3.  **Dynamic Schema Generation**: How to parse YAML configurations and use Python's `pydantic` library to generate data validation models on the fly.
4.  **End-to-End LLM Application Architecture**: How to tie everything together into a responsive, real-time web application using **Streamlit** and **LangChain**.

---

## 🚀 How It Works Under the Hood

This project is not just a chatbot; it is a pipeline generator. Here is the step-by-step lifecycle of what happens when you use the app:

### Phase 1: The Request (Vague to Structured)
1.  You type a vague, natural language idea into the left pane (e.g., *"Read an email and extract a summary, priority level, and if it requires an action"*).
2.  The application takes your messy input and feeds it to our local LLM (`gemma:2b`) using a strict **Meta-Prompt**.
3.  The LLM acts as an expert prompt engineer and outputs a highly structured JSON object representing a rigorous AI task. 

### Phase 2: Dynamic Instantiation (JSON to Python Code)
1.  The app receives this JSON output and saves it permanently to the `prompts/` directory as a `.yaml` file.
2.  The application dynamically reads the schema rules from the YAML file and uses Python's `pydantic.create_model()` to assemble a strict data validation class in memory—without you writing a single line of backend code! 

### Phase 3: Execution (Testing the Workflow)
1.  On the right pane, you paste sample raw data (like an actual messy customer email).
2.  When you execute the pipeline, LangChain runs the sample data through the generated prompt template.
3.  The response is intercepted by the `PydanticOutputParser`, which validates the LLM's output against our dynamic schema. If it matches, the clean JSON payload is delivered.

---

## 📂 Project Structure

```
├── app.py                  # The main Streamlit dashboard application (Start here!)
├── requirements.txt        # Python dependencies (Streamlit, LangChain, Pydantic, etc.)
│
├── prompts/                # Persisted storage for your generated YAML workflows
│   ├── system_prompt.yaml  # A sample structural output template
```

---

## 🛠️ Quick Start Guide

This application relies on `langchain-ollama` to run models locally and completely free without needing API keys. We are targeting the lightweight `gemma:2b` model by default.

### Prerequisites

1.  **Install Python 3.9+**
2.  **Install Ollama**: Download and install [Ollama](https://ollama.com/) for your operating system.
3.  **Pull the Model**: Open your terminal and run the following command to download the model we use for inference:
    ```bash
    ollama run gemma:2b
    ```
    *(You can close the prompt that opens up after it finishes downloading. Just ensure Ollama is running in the background).*

### Running the Project

1.  **Install Dependencies**: Navigate to the directory containing this README and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Streamlit Server**: Launch the interactive web user interface.
    ```bash
    streamlit run app.py
    ```

3.  **Explore the Dashboard**: 
    - Your default web browser will open (usually to `http://localhost:8501`).
    - Use the **left column** to describe an extraction task and generate a deterministic YAML template.
    - Use the **right column** to test your new custom AI agent with raw text to see the magic of structured extraction!

---

## 🧠 Assignment for Students

Want to challenge yourself? Try implementing the following features into `app.py`:

*   **Model Selection**: Add a Streamlit dropdown menu to let the user select between different local Ollama models (e.g., `llama3`, `mistral`) instead of hardcoding `gemma:2b`.
*   **Validation Fallbacks**: Right now, if the LLM hallucinates an invalid JSON structure, the app errors out. Can you wrap the execution in a LangChain `RetryOutputParser` so the LLM gets a second chance to fix its mistake?
*   **Database Integration**: Connect the parsed JSON outputs to a simple SQLite database so the structured data is actually saved for later analysis.
