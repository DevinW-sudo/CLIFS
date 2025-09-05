import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"   # stop torch.compile attempts
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ.pop("TORCH_LOGS", None)

import torch
torch.set_float32_matmul_precision('high')  # set high precision for matmul
import pandas as pd
from clifs import masked_lm_identity_fusion as mlmif
from clifs import unquestioning_affiliation_index, violence_risk_index_lite_clifs #, random_forest
from clifs import rag_fusion_classification as rag
from tqdm import tqdm
from clifs.uai_codes import *
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import joblib
import nltk
import getpass
from openai import OpenAI
import numpy as np
from collections import Counter
import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True
import logging, warnings
for name in ("torch._dynamo", "torch._inductor", "torch.overrides"):
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", module="torch._inductor")
warnings.filterwarnings("ignore", module="torch.overrides")


all_other_label_map = {0 : "high", 1 : "low", 2: "medium"}
all_other_label_map_inv = {v: k for k, v in all_other_label_map.items()}

global device, sbert_model, mbert_fine_tuned, clifs_model, clifs_model_rf, sbert_rf_model
global mbert_base, tokenizer, nlp
global rag_model_r1, rag_model_4o
global low_text, medium_text, high_text
global docs_df

# set all the above as None
for var in ['device', 'sbert_model', 'mbert_fine_tuned', 'clifs_model', 'clifs_model_rf',
            'sbert_rf_model', 'mbert_base', 'tokenizer', 'nlp',
            'rag_model_r1', 'rag_model_4o',
            'low_text', 'medium_text', 'high_text',
            'docs_df']:
    globals()[var] = None

def obtain_mlmif_all(df, known_groups, device=None, column='write'):
    global mbert_base, tokenizer, nlp
    if device is None:
        device = globals()['device']
    # obtain the mlmif scores
    # from the base Modern BERT model
    KNOWN_GROUPS = known_groups
    df['ItoT'] = df[column].apply(lambda x: mlmif.compute_identity_to_target_score(x, KNOWN_GROUPS, mbert_base, tokenizer, nlp, device))
    df['TtoI'] = df[column].apply(lambda x: mlmif.compute_target_to_identity_score(x, KNOWN_GROUPS, mbert_base, tokenizer, nlp, device))
    df['KtoT'] = df[column].apply(lambda x: mlmif.compute_kin_fusion_score(x, KNOWN_GROUPS, mbert_base, tokenizer, nlp, device))
    df['mlmif'] = df.apply(lambda x: mlmif.compute_fusion_proximity_score(x['ItoT'], x['TtoI']), axis=1)
    return df

def obtain_mlmif(text, known_groups, device=None):
    if device is None:
        device = globals()['device']
    # obtain the mlmif scores
    # from the base Modern BERT model
    KNOWN_GROUPS = known_groups
    ItoT = mlmif.compute_identity_to_target_score(text, KNOWN_GROUPS, mbert_base, tokenizer, nlp, device)
    TtoI = mlmif.compute_target_to_identity_score(text, KNOWN_GROUPS, mbert_base, tokenizer, nlp, device)
    KtoT = mlmif.compute_kin_fusion_score(text, KNOWN_GROUPS, mbert_base, tokenizer, nlp, device)
    mlmif_score = mlmif.compute_fusion_proximity_score(ItoT, TtoI)
    return ItoT, TtoI, KtoT, mlmif_score
   
def obtain_class_probs(text, model=None, tokenizer=None):
    if model is None:
        model = globals()['mbert_fine_tuned']
    if tokenizer is None:
        tokenizer = globals()['tokenizer']
    m_device = getattr(model, "device", next(model.parameters()).device)
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    inputs = {k: v.to(m_device) for k, v in inputs.items()}
    model.eval()
    
    with torch.inference_mode():
        outputs = model(**inputs)  
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

    return probabilities.cpu().numpy().flatten()
 
def obtain_class_probs_all(df, model=None, tokenizer=None):
    if model is None:
        model = globals()['mbert_fine_tuned']
    if tokenizer is None:
        tokenizer = globals()['tokenizer']
    # obtain the class probabilities
    # from fine-tuned Modern BERT model
    df['class_probabilities'] = df['write'].apply(obtain_class_probs, model=model, tokenizer=tokenizer)
    return df

def obtain_sbert_embeddings(text, model=None):
    if model is None:
        model = globals()['sbert_model']
    embedding = model.encode(text)
    return embedding

def obtain_uai_scores(text):
    uai_aff = unquestioning_affiliation_index.calculate_word_counts(text, affiliation)
    uai_cp = unquestioning_affiliation_index.calculate_word_counts(text, cogproc)
    nUAI = unquestioning_affiliation_index.calculate_nuai(uai_aff, uai_cp) # naive UAI
    return uai_aff, uai_cp, nUAI

def obtain_vri_scores(text):
    vri_dict = violence_risk_index_lite_clifs.process_text(text)
    vri_fusion = vri_dict['fusion']
    vri_identification = vri_dict['identification']
    return vri_fusion, vri_identification
    
def obtain_sbert_embedding_all(df, model=None):
    if model is None:
        model = globals()['sbert_model']
    # obtain the SBERT embeddings
    # from the all-mpnet-base-v2 model
    df['sbert_embeddings'] = df['write'].apply(obtain_sbert_embeddings, model=model)
    return df

def build_faiss_index(docs_df, model=None):
    if model is None:
        model = globals()['sbert_model']
    index = rag.build_faiss_index(docs_df, model)
    return index

def hard_vote(cl_rf, sb_rf, rag_4o, rag_r1):
    votes = [cl_rf, sb_rf, rag_4o, rag_r1]
    counter = Counter(votes)
    most_common = counter.most_common()
    
    top_count = most_common[0][1]
    top_candidates = [label for label, count in most_common if count == top_count]
    
    # Tie-breaker: choose based on fixed priority order (e.g., high > low > medium)
    priority = {'high': 3, 'low': 2, 'medium': 1}
    top_candidates.sort(key=lambda x: priority[x], reverse=True)
    
    y_pred = top_candidates[0]
    return y_pred

def load_models():
    global device, sbert_model, mbert_fine_tuned, clifs_model, clifs_model_rf, sbert_rf_model
    global mbert_base, tokenizer, nlp
    global rag_model_r1, rag_model_4o
    global low_text, medium_text, high_text
    global docs_df
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train set
    aug_train = '../data/ap_study1_augmented_finegrain_train.csv'
    # for rag
    docs_df = pd.read_csv(aug_train)
        
    # lowest scoring samples from each train set
    min_score = min(docs_df['fusion'])
    low_text = docs_df[docs_df['fusion'] == min_score].iloc[0]['write']

    # most middle scoring samples from each train set
    middle_score = docs_df['fusion'].median()
    medium_text = docs_df[docs_df['fusion'] == middle_score].iloc[0]['write']

    # highest scoring samples from each train set
    max_score = max(docs_df['fusion'])
    high_text = docs_df[docs_df['fusion'] == max_score].iloc[0]['write']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    sbert_model.to(device)
    mbert_fine_tuned = AutoModelForSequenceClassification.from_pretrained("../models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain")
    mbert_fine_tuned.to(device)
    mbert_fine_tuned.eval()
    clifs_model = joblib.load('../models/best_rf/best_model_rf_aug.joblib')
    sbert_rf_model = joblib.load('../models/best_rf/sbert_classification_best_model_augmented.joblib')
    rag_model_r1 = 'deepseek-reasoner'
    rag_model_4o = 'gpt-4o'
    mbert_base, tokenizer, nlp = mlmif.load_model_nlp_and_tokenizer(device=device)
    
    # assert none of the global variables are None
    assert device is not None, "Device is not set."
    assert clifs_model is not None, "CLIFS model is not loaded."
    assert sbert_model is not None, "SBERT model is not loaded."
    assert mbert_fine_tuned is not None, "ModernBERT model is not loaded."
    assert mbert_base is not None, "ModernBERT base model is not loaded."
    assert tokenizer is not None, "ModernBERT tokenizer is not loaded."
    assert nlp is not None, "NLP model is not loaded."
    assert rag_model_r1 is not None, "RAG model R1 is not loaded."
    assert rag_model_4o is not None, "RAG model 4o is not loaded."
    assert docs_df is not None, "Docs DataFrame is not loaded."
    assert low_text is not None, "Low text is not loaded."
    assert medium_text is not None, "Medium text is not loaded."
    assert high_text is not None, "High text is not loaded."
    
def load_model():
    global device, clifs_model, sbert_model, mbert_fine_tuned
    global mbert_base, tokenizer, nlp
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clifs_model = joblib.load('../models/best_rf/best_model_rf_aug.joblib')
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    sbert_model.to(device)
    mbert_fine_tuned = AutoModelForSequenceClassification.from_pretrained("../models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain")
    mbert_fine_tuned.to(device)
    mbert_fine_tuned.eval()
    mbert_base, tokenizer, nlp = mlmif.load_model_nlp_and_tokenizer(device=device)
    
    # assert none of the global variables are None
    assert device is not None, "Device is not set."
    assert clifs_model is not None, "CLIFS model is not loaded."
    assert sbert_model is not None, "SBERT model is not loaded."
    assert mbert_fine_tuned is not None, "ModernBERT model is not loaded."
    assert mbert_base is not None, "ModernBERT base model is not loaded."
    assert tokenizer is not None, "ModernBERT tokenizer is not loaded."
    assert nlp is not None, "NLP model is not loaded."

def load_regression_model():
    global device, clifs_model_r, sbert_model, mbert_fine_tuned
    global mbert_base, tokenizer, nlp
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clifs_model_r = joblib.load('../models/best_rf/augmented_regression_best.joblib')
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    sbert_model.to(device)
    mbert_fine_tuned = AutoModelForSequenceClassification.from_pretrained("../models/best_mbert_model/best_mbert_model/modern_BERT_fusion_augmented_data_finegrain")
    mbert_fine_tuned.to(device)
    mbert_fine_tuned.eval()
    mbert_base, tokenizer, nlp = mlmif.load_model_nlp_and_tokenizer(device=device)
    
    # assert none of the global variables are None
    assert device is not None, "Device is not set."
    assert clifs_model_r is not None, "CLIFS regression model is not loaded."
    assert sbert_model is not None, "SBERT model is not loaded."
    assert mbert_fine_tuned is not None, "ModernBERT model is not loaded."
    assert mbert_base is not None, "ModernBERT base model is not loaded."
    assert tokenizer is not None, "ModernBERT tokenizer is not loaded."
    assert nlp is not None, "NLP model is not loaded."

def clifs(df, known_groups=[], ensemble=False, regression=False, save_path='../clifs_predictions.csv'):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cwd = os.getcwd()
    # get the path to this directory
    clifs_dir = os.path.dirname(os.path.abspath(__file__))
    # go up 3 directories 
    clifs_dir = os.path.dirname(os.path.dirname(os.path.dirname(clifs_dir)))
    # check if "exec_dir" exists and if not, create it
    exec_path = os.path.join(clifs_dir, 'exec_dir')
    if not os.path.exists(exec_path):
        os.makedirs(exec_path)
    # set the current working directory to the exec_dir
    os.chdir(exec_path)
    
    nltk.download('punkt')
    if ensemble:
        load_models()
        regression = False
    
        ds_key = getpass.getpass("Enter your DeepSeek API key: ")
        client_ds = OpenAI(
            # base_url="https://openrouter.ai/api/v1",
            base_url="https://api.deepseek.com",
            # obtain key from pop up
            api_key=ds_key
        )
        
        oai_key = getpass.getpass("Enter your OpenAI API key: ")
        client_oai = OpenAI(
            api_key=oai_key
        )

        print('Building FAISS index...')
        index = build_faiss_index(docs_df, sbert_model)
        
    elif regression:
        load_regression_model()
    else:
        load_model()
    
    # save every 10
    save_every = 10
        
    pbar = tqdm(total=len(df), desc="Processing texts")
    for counter, (idx, row) in enumerate(df.iterrows()):
        # first check if row is already processed
        # does row have column
        if row.get('sbert_rf') is not None and pd.notna(row.get('sbert_rf')):
            pbar.update(1)
            continue
        
        pbar.set_description(f"Processing text {counter+1}/{len(df)}")
        text = row['text']

        pbar.set_description(f"Extracting features for text {counter+1}/{len(df)}")
        # --- Extract features from the text ---
        # obtain the mlmif scores
        ItoT, TtoI, KtoT, mlmif_score = obtain_mlmif(text, known_groups)
        # obtain ModernBERT class probabilities
        class_probs = obtain_class_probs(text)
        # obtain SBERT embeddings
        sbert_embeddings = obtain_sbert_embeddings(text)
        # obtain UAI scores
        uai_aff, uai_cp, nUAI = obtain_uai_scores(text)
        # obtain VRI scores
        vri_fusion, vri_identification = obtain_vri_scores(text)
        
        # --- Flatten and ensure everything is 1D ---
        mlmif_feats = np.array([ItoT, TtoI, KtoT, mlmif_score])
        uai_feats = np.array([uai_aff, uai_cp, nUAI])
        vri_feats = np.array([vri_fusion, vri_identification])
        
        # --- Stack in consistent order ---
        feature_vector = np.hstack([
            sbert_embeddings,           # 768 dims
            class_probs,                # 3 dims
            mlmif_feats,                # 4 dims
            uai_feats,                  # 3 dims
            vri_feats                   # 2 dims
        ])
        
        
        if ensemble:
            # --- Make prediction with CogLing RF ---
            pbar.set_description(f"Making prediction with CogLing RF for text {counter+1}/{len(df)}")
            clifs_prediction = clifs_model.predict(feature_vector.reshape(1, -1))[0]
            
            # --- Make prediction with SBERT RF ---
            pbar.set_description(f"Making prediction with SBERT RF for text {counter+1}/{len(df)}")
            sbert_embeddings = np.array(sbert_embeddings).reshape(1, -1)
            sbert_rf_prediction = sbert_rf_model.predict(sbert_embeddings)[0]
            
            # --- Make prediction with RAG 4o ---
            pbar.set_description(f"Making prediction with RAG OpenAI gpt-4o for text {counter+1}/{len(df)}")
            rag_4o_prediction = rag.classify_text_rag(
                                    text=text,
                                    low_text=low_text,
                                    medium_text=medium_text,
                                    high_text=high_text,
                                    client=client_oai,
                                    index=index,
                                    model=sbert_model,
                                    rag_model=rag_model_4o,
                                    docs_df=docs_df
                                )
            
            # --- Make prediction with RAG R1 ---
            pbar.set_description(f"Making prediction with RAG DeepSeek R1 for text {counter+1}/{len(df)}")
            rag_r1_prediction = rag.classify_text_rag(
                                    text=text,
                                    low_text=low_text,
                                    medium_text=medium_text,
                                    high_text=high_text,
                                    client=client_ds,
                                    index=index,
                                    model=sbert_model,
                                    rag_model=rag_model_r1,
                                    docs_df=docs_df
                                )
            
            vote = hard_vote(
                cl_rf=clifs_prediction,
                sb_rf=sbert_rf_prediction,
                rag_4o=rag_4o_prediction,
                rag_r1=rag_r1_prediction
            )
            
            # --- Save predictions ---
            df.at[idx, 'sbert_rf'] = sbert_rf_prediction
            df.at[idx, 'clifs'] = clifs_prediction
            df.at[idx, 'rag_4o'] = rag_4o_prediction
            df.at[idx, 'rag_r1'] = rag_r1_prediction
            df.at[idx, 'clifs_fusion'] = vote
            df.at[idx, 'clifs_fusion_numeric'] = all_other_label_map_inv[vote]
            
        elif regression:
            # --- Make prediction with CLIFS regression model ---
            pbar.set_description(f"Making prediction with CLIFS regression model for text {counter+1}/{len(df)}")
            clifs_regression_prediction = clifs_model_r.predict(feature_vector.reshape(1, -1))[0]
            
            # --- Save predictions ---
            df.at[idx, 'clifs_fusion_r'] = clifs_regression_prediction
            
        else:
            # --- Make prediction with CLIFS ---
            pbar.set_description(f"Making prediction with CLIFS for text {counter+1}/{len(df)}")
            clifs_prediction = clifs_model.predict(feature_vector.reshape(1, -1))[0]
            
            # --- Save predictions ---
            df.at[idx, 'clifs'] = clifs_prediction
            df.at[idx, 'clifs_fusion_numeric'] = all_other_label_map_inv[clifs_prediction]
        
        # Save every 10
        if (counter + 1) % save_every == 0:
            # use cwd to save the file
            # change back to the original directory
            os.chdir(cwd)
            df.to_csv(save_path, index=False)
            # change back to the exec_dir
            os.chdir(exec_path)
        
        pbar.update(1)
    pbar.close()
    # change back to the original directory
    os.chdir(cwd)
    # save the predictions to a csv file
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
    return df