import pandas as pd
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from openai import OpenAI
import os

def search(query, index, model, docs_df, number_of_results=5):
    """
    Given an input text (query), retrieve the top k similar training examples.
    Returns a list of dictionaries containing the text, label, and distance.
    """
    query_embed = model.encode(query)
    distances, indices = index.search(np.float32([query_embed]), number_of_results)
    
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            'text': docs_df.iloc[idx]['write'],
            'label': docs_df.iloc[idx]['label'],
            'distance': distances[0][i]
        }
        results.append(result)
    return results

# ------------------------------
# DeepSeek or OpenAI API call
# ------------------------------
def get_response(client, messages, model="deepseek-reasoner"):
    max_retries = 3  # Total retries (initial attempt + 2 retries)
    retries = 0
    timeout = 180  # 3 minutes

    while retries < max_retries:
        try:
            messages = messages
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
                temperature=0,
                timeout=timeout
            )
            
            raw_output = response.choices[0].message.content.strip().lower()
            
            # Check if the output is strictly "low", "medium", or "high"
            if raw_output in {"low", "medium", "high"}:
                return raw_output
            else:
                # If invalid, check if it contains a valid label (fallback)
                if "high" in raw_output:
                    return "high"
                elif "low" in raw_output:
                    return "low"
                elif "medium" in raw_output:
                    return "medium"
                else:
                    # If still invalid, retry with a stricter instruction
                    if retries == 0: # Only retry once
                        print(f"Invalid response: '{raw_output}'. Retrying with correction...")
                        correction_prompt = (
                            f"Previous response: '{raw_output}'\n"
                            "This is invalid. You must output **only** one of these labels: [low, medium, high].\n"
                            f"Try again: \n\n"
                        )
                        messages.append({"role": "assistant", "content": raw_output})
                        messages.append({"role": "user", "content": correction_prompt})
                        retries += 1
                        continue
                    else:
                        print(f"Failed to get a valid response after retries. Returning default 'medium'.")
                        return "medium"
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries})")
            time.sleep(5)
    
    print("Failed to get a response after multiple attempts.")
    return "medium"  # Fallback default

# -------------------------------------
# RAG-based classification function
# -------------------------------------
def classify_text_rag(text, low_text, medium_text, high_text, client, index, model, rag_model, docs_df):
    # Retrieve the top 5 most similar training examples.
    retrieved_examples = search(text, index, model, docs_df, number_of_results=5)
    
    # Start with the system instruction.
    messages = [
        {
            "role": "system",
            "content": (
                f"""You are a text classifier that determines the level of identity fusion in a given text. Identity fusion is when an individual's personal identity becomes strongly intertwined with their target's identity. 

Based on Swann et al. (2024), identity fusion is a psychological state in which an individual’s personal identity becomes deeply intertwined with a target—be it a group, leader, value, or cause—resulting in porous boundaries between the self and that target. This fusion creates a powerful reciprocal bond where personal agency is channeled into extreme, pro-target behavior, with the individual experiencing a profound “sense of oneness” that can motivate costly and self-sacrificial actions in defense of the fusion target.

In this task, label the text as:
  - "low": Minimal fusion between individual and target identity. Low fusion is marked by a clear separation between the self and the target, so the individual shows little behavioral commitment to the target.
  - "medium": Moderate fusion between individual and target identity. Medium fusion reflects a moderate integration where the personal self overlaps with the target enough to inspire occasional support without overwhelming personal autonomy.
  - "high": Strong fusion; the individual's identity is almost completely merged with the target's identity. High fusion is characterized by an intense, nearly inseparable merging of identity with the target, driving individuals to engage in extreme, self-sacrificial actions for its sake.

Below are a three examples:

Example 1 (Lowest Scoring - low):
Classify the following text into [low, medium, high]:
Text: "{low_text}"
Output only the label, nothing else.
Label: low

Example 2 (Most Middle Scoring - medium):
Classify the following text into [low, medium, high]:
Text: "{medium_text}"
Output only the label, nothing else.
Label: medium

Example 3 (Highest Scoring - high):
Classify the following text into [low, medium, high]:
Text: "{high_text}"
Output only the label, nothing else.
Label: high\n\n"""
            )
        }
    ]
    
    # Append retrieved examples with their classifications.
    for ex in retrieved_examples:
        messages.append({
            "role": "user",
            "content": f"""Classify the following text into [low, medium, high]:
Text: "{ex['text']}"
Output only the label, nothing else.
Label:"""
        })
        messages.append({
            "role": "assistant",
            "content": ex["label"]
        })
    
    # Finally, add the target text for which a classification is needed.
    messages.append({
            "role": "user",
            "content": f"""Classify the following text into [low, medium, high]:
Text: "{text}"
Output only the label, nothing else.
Label:"""
        })
    
    # Get the response from the client using the assembled messages.
    classification = get_response(messages=messages, client=client, model=rag_model)
    return classification


def build_faiss_index(docs_df, model):
    # --------------------------------------
    # Build FAISS index with SBERT embeddings
    # --------------------------------------
    # check if faiss index already exists
    index_path = '../data/faiss_index.index'
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print("FAISS index loaded from disk.")
    else:
        # Get embeddings for the training set texts
        docs_texts = docs_df['write'].tolist()
        docs_embeddings = model.encode(docs_texts)
        embeds = np.array(docs_embeddings)
        dim = embeds.shape[1]
        
        # Create a FAISS index and add the embeddings
        index = faiss.IndexFlatL2(dim)
        index.add(np.float32(embeds))
        
        # Save the index to disk
        faiss.write_index(index, index_path)
        print("FAISS index saved to disk.")
        
    return index
