# ========================
# SETUP: Hugging Face Transformers and Environment
# ========================

import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# Hugging Face setup — use your own token for access to huggingface llama
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"

login(HUGGINGFACE_TOKEN)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HUGGINGFACE_TOKEN,
    device_map="auto",
    torch_dtype="auto"
)

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# ========================
# Load Discord Data
# ========================
input_folder = 'filtered_JSON' #have filtered_JSON folder in the same directory as this script
output_path = 'LLaMA_emotion_sentiment_ALL.xlsx'

def get_all_documents(folder):
    all_docs = []
    files =  {} #for looking at specific files which we don't need anymore

    for fname in os.listdir(folder):
        #if fname not in files:
            #continue
        with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
            channel = data.get("channel", "Unknown")
            channel_name = channel.get("name") if isinstance(channel, dict) else channel

            for msg in data.get("messages", []):
                content = msg.get("content", "")
                if content and "joined the server" not in content.lower():
                    all_docs.append({
                        "channel": channel_name,
                        "user": msg.get("author", {}).get("name", "Unknown"),
                        "timestamp": msg.get("timestamp", "Unknown"),
                        "content": content
                    })
    return all_docs

df = pd.DataFrame(get_all_documents(input_folder))
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['timestamp'] = df['timestamp'].dt.strftime('%m/%d/%Y %H:%M')

# ========================
# PROMPTS
# ========================

EMOTION_PROMPT = """
You are an assistant that analyzes Discord messages for emotion.
Reply with ONLY one of these emotions:
Emotion: [Gratitude, Ingratitude, Respect, Disrespect, Excitement, Forgiveness, Unforgiveness, Arrogance, Humility, Eagerness, Reluctance, Helpfulness, Frustration, Confusion]

Example:
Message: "Thanks for your help!"
Emotion: Gratitude

Now analyze this message:
Message: "{message}"
Emotion:
"""

SENTIMENT_PROMPT = """
You are an assistant that analyzes Discord messages for sentiment.
Reply with ONLY one of the following:
Sentiment: [Positive, Neutral, Negative, None]

Example:
Message: "Thanks for your help!"
Sentiment: Positive

Now analyze this message:
Message: "{message}"
Sentiment:
"""

# ========================
# EMOTION CLASSIFICATION
# ========================

def classify_with_prompt(messages, prompt_template, pattern, label_name="emotion"):
    results = []
    invalids = []

    for i, msg in enumerate(tqdm(messages, desc=f"Classifying {label_name}")):
        prompt = prompt_template.format(message=msg)
        try:
            output = generator(prompt, max_new_tokens=50)[0]['generated_text']
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results.append(match.group(1).lower())
            else:
                results.append("invalid")
                invalids.append(i)
        except Exception as e:
            print(f"Error on {label_name} index {i}: {e}")
            results.append("error")
            invalids.append(i)

    return results, invalids

def reclassify_until_valid(df, column_name, messages, prompt_template, pattern, label_name="emotion", max_retries=300):
    for attempt in range(max_retries):
        invalid_indices = df[df[column_name] == "invalid"].index
        if len(invalid_indices) == 0:
            print(f"All {label_name} entries valid after {attempt} retry pass(es).")
            break
        print(f"🔁 Retry {attempt + 1}/{max_retries}: {len(invalid_indices)} invalid {label_name} entries remaining...")
        
        for i in tqdm(invalid_indices, desc=f"Retrying {label_name}, pass {attempt+1}"):
            prompt = prompt_template.format(message=messages[i])
            try:
                output = generator(prompt, max_new_tokens=50)[0]['generated_text']
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    df.at[i, column_name] = match.group(1).lower()
            except Exception as e:
                print(f"Error retrying index {i}: {e}")
    else:
        print(f"❗ Reached max retries for {label_name}. {len(df[df[column_name] == 'invalid'])} still invalid.")
# EMOTION classification
emotion_pattern = r"Emotion:\s*(Gratitude|Ingratitude|Respect|Disrespect|Excitement|Forgiveness|Unforgiveness|Arrogance|Humility|Eagerness|Reluctance|Helpfulness|Frustration|Confusion)"
df["llama_emotion"], _ = classify_with_prompt(df["content"], EMOTION_PROMPT, emotion_pattern)
reclassify_until_valid(df, "llama_emotion", df["content"], EMOTION_PROMPT, emotion_pattern, label_name="emotion")

# SENTIMENT classification
sentiment_pattern = r"Sentiment:\s*(Positive|Neutral|Negative)"
df["llama_sentiment"], _ = classify_with_prompt(df["content"], SENTIMENT_PROMPT, sentiment_pattern, label_name="sentiment")
reclassify_until_valid(df, "llama_sentiment", df["content"], SENTIMENT_PROMPT, sentiment_pattern, label_name="sentiment")

# ========================
# SAVE RESULTS
# ========================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_excel(output_path, index=False)

# Summary sheets
emotion_counts = df["llama_emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]

sentiment_counts = df["llama_sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
    emotion_counts.to_excel(writer, sheet_name="Emotion Counts", index=False)
    sentiment_counts.to_excel(writer, sheet_name="Sentiment Counts", index=False)

print(f"Results saved to: {output_path}")
