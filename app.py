
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="AI Quiz Generator", page_icon="📝")

st.title("📝 AI Quiz Generator")
st.write("Generate questions from any paragraph using AI.")

@st.cache_resource
def load_model():
    model_name = "Gulnisa/ai-quiz-generator"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

context = st.text_area("Enter your paragraph:", height=200)

if st.button("Generate Quiz"):
    if context.strip():
        inputs = tokenizer(
            "generate question: " + context,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            num_return_sequences=5
        )

        st.subheader("Generated Questions")

        for i, output in enumerate(outputs):
            question = tokenizer.decode(
                output,
                skip_special_tokens=True
            )
            st.write(f"{i+1}. {question}")
    else:
        st.warning("Please enter a paragraph first.")





