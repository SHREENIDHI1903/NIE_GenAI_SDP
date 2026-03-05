import streamlit as st
import yaml
import json
import os
from typing import Dict, Any, Type
from pydantic import BaseModel, Field, create_model
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ---- CONFIGURATION ----
st.set_page_config(page_title="Deterministic Workflow Builder", layout="wide")
# Use absolute path relative to the script to avoid WinError 433
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)

# ---- META-PROMPT ----
META_PROMPT_TEMPLATE = """Task: Convert a vague request into a DENSE, deterministic LLM prompt configuration.

Rules for the generated 'template':
1. Explicitly list every key from the 'schema'.
2. Provide a strict extraction rule for each key.
3. Command the model to use DOUBLE QUOTES for all keys and skip all preamble.

Example:
Request: "Analyze sentiment and priority"
Result JSON: {{
  "name": "sentiment_priority",
  "description": "Analyzes tone and urgency",
  "template": "Analyze: {{input_data}}. Rules: Extract 'tone' (Happy/Sad), 'score' (0-1), and 'priority' (High/Low). You MUST use double quotes for all keys. Respond ONLY with JSON. {{format_instructions}}",
  "schema": {{"tone": "string", "score": "float", "priority": "string"}}
}}

---
Request: "{vague_prompt}"
Result JSON: """

import re

# ---- HELPER FUNCTIONS ----
def clean_json_response(text: str) -> str:
    """Robustly extracts and repairs JSON from LLM output."""
    text = text.strip()
    
    def repair_json(json_str: str) -> str:
        # 1. Fix unquoted keys at start of object or after comma
        repaired = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', json_str)
        # 2. Fix unquoted keys at start of line (if no comma/brace)
        repaired = re.sub(r'^(\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', repaired, flags=re.MULTILINE)
        # 3. Fix missing commas between lines
        # Look for: "key": value \n "next_key":
        repaired = re.sub(r'("[^"]+"\s*:\s*[^,{}\n]+)\n\s*("[^"]+"\s*:)', r'\1,\n\2', repaired)
        return repaired

    start_idx = text.find('{')
    while start_idx != -1:
        balance = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '{': balance += 1
            if text[i] == '}': balance -= 1
            if balance == 0:
                end_idx = i
                break
        
        if end_idx != -1:
            candidate = text[start_idx:end_idx + 1]
            try:
                # Try raw first
                json.loads(candidate)
                return candidate
            except:
                # Try repair
                try:
                    repaired = repair_json(candidate)
                    json.loads(repaired)
                    return repaired
                except: pass
        
        start_idx = text.find('{', start_idx + 1)
    return text.strip()

@st.cache_resource
def get_llm():
    # Use a low temperature for strict generations
    return OllamaLLM(model="gemma:2b", temperature=0.1)


def generate_yaml_config(vague_text: str) -> dict:
    llm = get_llm()
    # Completion style for meta-generation
    final_prompt = META_PROMPT_TEMPLATE.format(vague_prompt=vague_text)
    
    response_text = llm.invoke(final_prompt)
    cleaned_text = clean_json_response(response_text)
    
    # Log for debugging if parsing fails
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        st.error("Severe Model Error: Could not extract valid JSON from generator.")
        st.code(response_text)
        return None

def create_dynamic_pydantic_model(model_name: str, schema_dict: dict) -> Type[BaseModel]:
    """Dynamically creates a Pydantic Model from a dictionary schema mapping."""
    fields = {}
    for key, description in schema_dict.items():
        # Defaulting simple everything to string, but giving it the LLM description for context
        fields[key] = (str, Field(description=str(description)))
    
    # Create the model dynamically
    return create_model(model_name, **fields)

def execute_workflow(template_str: str, pydantic_model: Type[BaseModel], input_data: str) -> dict:
    llm = get_llm()
    
    # Extract expected keys
    expected_keys = list(pydantic_model.model_fields.keys())
    
    # 1. Final Reinforcement: Prepend a strict key list to the input
    key_reinforcement = f"Strict Requirement: Your JSON MUST use double quotes for all keys: {expected_keys}\n\n"

    
    # 2. Structure as a Completion Task with JSON opening brace forcing
    # We end the prompt with '{' to guide the 2B model to start the keys immediately
    full_prompt_text = f"Task Instructions:\n{template_str}\n\n{key_reinforcement}Input Content:\n\"\"\"\n{input_data}\n\"\"\"\n\nExtracted Result JSON:\n{{"
    
    # Simplify format instructions
    schema_json = json.dumps(pydantic_model.model_json_schema(), indent=2)
    format_instructions = f"Output ONLY a raw JSON object matching this schema:\n{schema_json}"
    
    try:
        # Pre-calculate payload replacements
        final_prompt = full_prompt_text.replace("{input_data}", input_data).replace("{format_instructions}", format_instructions)
        
        # Get raw response
        response_suffix = llm.invoke(final_prompt)
        
        # We manually add the starting brace back since we forced it in the prompt
        raw_response = "{" + response_suffix
        cleaned_text = clean_json_response(raw_response)
        
        # Check if we actually have JSON before parsing
        if '{' not in cleaned_text or '}' not in cleaned_text:
            return {"error": "LLM refused to output JSON and returned conversation instead.", "raw_output": raw_response}

        # Parse the JSON
        data = json.loads(cleaned_text)
        
        # ---- FUZZY KEY MAPPING ----
        final_data = {}
        hallucinated_keys = list(data.keys())
        
        for key in expected_keys:
            if key in data:
                final_data[key] = data[key]
            else:
                match_found = False
                for h_key in hallucinated_keys:
                    if key.lower() in h_key.lower() or h_key.lower() in key.lower():
                        final_data[key] = data[h_key]
                        match_found = True
                        break
                if not match_found:
                    final_data[key] = None
        
        response_model = pydantic_model(**final_data)
        return response_model.model_dump()
    except Exception as e:
        return {"error": str(e), "raw_output": raw_response if 'raw_response' in locals() else "No output"}

# ---- UI LAYOUT ----
st.title("Deterministic Workflow Builder 🛠️")
st.markdown("Instantly convert vague ideas into strict, Pydantic-validated data pipelines.")

col1, col2 = st.columns([1, 1])

# STATE VARIABLES
if 'generated_config' not in st.session_state:
    st.session_state['generated_config'] = None
if 'model_class' not in st.session_state:
    st.session_state['model_class'] = None

with col1:
    st.header("1. Define the Workflow")
    vague_input = st.text_area("Enter your vague prompt or objective:", height=200, 
                               placeholder="e.g. Read an email, give me a short summary, a priority (High/Medium/Low), and a boolean if it's urgent.")
    
    if st.button("Generate Deterministic Workflow", type="primary"):
        with st.spinner("Engineering prompt architecture..."):
            config = generate_yaml_config(vague_input)
            if config:
                st.session_state['generated_config'] = config
                
                # Save to file system as well
                filepath = os.path.join(PROMPTS_DIR, f"{config.get('name', 'dynamic_prompt')}.yaml")
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, sort_keys=False)
                st.success(f"Saved configuration to `{filepath}`")
                
                # Generate the pydantic model based on the schema dict
                schema = config.get("schema", {"result": "The final output"})
                model_name = config.get("name", "DynamicOutput").replace("_", " ").title().replace(" ", "")
                st.session_state['model_class'] = create_dynamic_pydantic_model(model_name, schema)

    if st.session_state['generated_config']:
        st.subheader("Generated Configuration (YAML)")
        st.code(yaml.dump(st.session_state['generated_config'], sort_keys=False), language="yaml")

with col2:
    st.header("2. Test the Workflow")
    if not st.session_state['generated_config']:
        st.info("Generate a workflow on the left before testing it here.")
    else:
        st.write(f"**Loaded template:** `{st.session_state['generated_config'].get('name')}`")
        test_input = st.text_area("Enter sample input text to parse:", height=200, placeholder="Drop in the text you want the LLM to structure.")
        
        if st.button("Execute Pipeline", type="secondary"):
            with st.spinner("Running deterministic extraction..."):
                template_text = st.session_state['generated_config'].get('template')
                model_cls = st.session_state['model_class']
                
                result = execute_workflow(template_text, model_cls, test_input)
                
                st.subheader("Extraction Result")
                if "error" in result:
                    st.error("Validation Failed")
                    st.json(result)
                else:
                    st.success("Successfully Parsed!")
                    st.json(result)
