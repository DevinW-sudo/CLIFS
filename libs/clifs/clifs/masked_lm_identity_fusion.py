# This module computes four scores:
#   1. Identity→Target score: identity words replacing target words.
#   2. Target→Identity score: target words replacing identity words.
#   3. Fusion Proximity Score: The harmonic mean of the two directional scores.
#   4. Kin Fusion (Kin→Target) score: kin words replacing target words.

# The core idea is to mask target tokens in text (via NER and embedding augmented
# base word lists) and use a masked language model to compute how similar personal
# and target identity are directionally by computing how likely each representative
# set could fit in place of the other in an aggregate score. (metaphors)

# Equations:
#   - Directional score:
#       score_{x→y} = (1/M_y) * sum_{m=1}^{M_y} sum_{w_v in V_x} (P(w_v | context_m))^alpha 
  
#   - Fusion proximity score (harmonic mean):
#       f_{(I,T)} = (2 * S_{I→T} * S_{T→I}) / (S_{I→T} + S_{T→I})

#  - Kin Fusion score:
#       K_f = score_{K→T}

import torch
import spacy
from clifs import vri_codes_lite_clifs as vri_codes
import gensim.downloader as api
# Default candidate word lists (customize as needed)
IDENTITY_WORDS = set()
TARGETS = set()

IDENTITY_WORDS = {"i", "me", "my", "mine", "myself"}
# add fusion words (e.g. kin and shared blood) from vri_codes
KIN_IF_WORDS = vri_codes.fusion

COLLECTIVE_FIRST_PERSON_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}

TARGET_SEED_WORDS = { "team", "class", "club", "society", "squad", "gang", "band", "crew" }

TARGETS = TARGETS.union(TARGET_SEED_WORDS)

TARGETS_ARE_EXPANDED = False

KIN_IF_ARE_EXPANDED = False

def load_model_nlp_and_tokenizer(model_name='answerdotai/ModernBERT-base', 
                                 nlp_model='en_core_web_sm', device='cpu'):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = spacy.load(nlp_model)
    return model, tokenizer, nlp

def mask_entities(text, base_words, nlp, entity_labels=None):
    """
    Masks tokens in the given text based on entity detection and/or a list of base words.

    Parameters:
      text (str): The input text.
      base_words (list): List of words (case-insensitive) to mask if found in the text.
      nlp: A spaCy language model for NER.
      entity_labels (list): (Optional) List of spaCy entity labels to mask (e.g., ["ORG", "NORP", "GPE"]).

    Returns:
      str: The text with target words replaced by the mask token.
    """
    mask_token="[MASK]"
    doc = nlp(text)
    masked_text = text
    # Mask tokens detected by NER with matching entity labels.
    if entity_labels:
        for ent in doc.ents:
            if ent.label_ in entity_labels:
                masked_text = masked_text.replace(ent.text, mask_token)
    # Mask additional tokens if they appear in the base_words list.
    for token in doc:
        if token.text.lower() in [w.lower() for w in base_words]:
            masked_text = masked_text.replace(token.text, mask_token)
    return masked_text
  
def compute_score(masked_text, candidate_words, model, tokenizer, device, alpha=0.5):
    """
    Computes a score based on exponentiated probabilities of candidate words
    for each masked token.

    Equation:
        score_{x→y} = (1/M_y) * sum_{m=1}^{M_y} sum_{w_v in V_x} (P(w_v | context_m))^alpha 
    """
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs.input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    if mask_positions.numel() == 0:
        return 0.0
    
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)

    total_sum = 0.0
    for pos in mask_positions:
        token_probs = predictions[0, pos, :]
        inner_sum = 0.0
        for word in candidate_words:
            word_id = tokenizer.convert_tokens_to_ids(word)
            prob = token_probs[word_id].item()
            inner_sum += prob ** alpha
        total_sum += inner_sum

    M = len(mask_positions)
    if M == 0 or total_sum == 0.0:
        return 0.0
    avg_sum = total_sum / M
    score = avg_sum
    return score

  
def set_targets(target_base_words, threshold=0.8, topn=1000):
    """
    Expands the list of target_base_words by retrieving words from the gensim_model's vocabulary
    that have a cosine similarity >= threshold with any of the base words.
    
    Parameters:
      target_base_words (set): The initial set of target-related base words.
      threshold (float): The cosine similarity threshold to consider a word similar.
      topn (int): Maximum number of similar words to retrieve for each base word.
    
    Returns:
      A set of words containing the original seed words and any candidate words with similarity >= threshold.
    """
    global TARGETS, TARGETS_ARE_EXPANDED, COLLECTIVE_FIRST_PERSON_PRONOUNS
    gensim_model = api.load("glove-wiki-gigaword-100")
    expanded_targets = set(target_base_words)
    expanded_targets = expanded_targets.union(TARGETS)
    
    for base in target_base_words:
        try:
            similar_words = gensim_model.most_similar(base, topn=topn)
        except KeyError:
            continue
        
        for word, sim in similar_words:
            if sim >= threshold:
                expanded_targets.add(word)
            else:
                # most_similar returns words in descending order of similarity,
                # so if we fall below the threshold we can break early.
                break
    
    TARGETS = expanded_targets
    TARGETS.union(COLLECTIVE_FIRST_PERSON_PRONOUNS)
    TARGETS_ARE_EXPANDED = True
    
def set_kin_if_words(threshold=0.8, topn=1000):
    """
    Expands the list of kin_if_base_words by retrieving words from the gensim_model's vocabulary
    that have a cosine similarity >= threshold with any of the base words.
    
    Parameters:
      threshold (float): The cosine similarity threshold to consider a word similar.
      topn (int): Maximum number of similar words to retrieve for each base word.
      
    Returns:
      A set of words containing the original seed words and any candidate words with similarity >= threshold.
    """
    
    global KIN_IF_WORDS, KIN_IF_ARE_EXPANDED
    gensim_model = api.load("glove-wiki-gigaword-100")
    expanded_kin_if = set()
    expanded_kin_if = expanded_kin_if.union(KIN_IF_WORDS)
    
    for base in KIN_IF_WORDS:
        try:
            similar_words = gensim_model.most_similar(base, topn=topn)
        except KeyError:
            continue
        
        for word, sim in similar_words:
            if sim >= threshold:
                expanded_kin_if.add(word)
            else:
                # most_similar returns words in descending order of similarity,
                # so if we fall below the threshold we can break early.
                break
              
    KIN_IF_WORDS = expanded_kin_if
    KIN_IF_ARE_EXPANDED = True


def compute_identity_to_target_score(text, target_base_words, model, tokenizer, nlp, device, alpha=0.5):
    """
    Computes the Identity → Target score.
    
    This function expands target_base_words vocabulary (via set_targets with gensim),
    masks target-related words in the text (using spaCy NER and the expanded target words),
    and then computes the log-average probability that these masks are filled by identity-related
    candidate words (using the immutable DEFAULT_IDENTITY_WORDS).

    Parameters:
      text (str): The input text.
      target_base_words (set): List of base target-related words.
      model, tokenizer, device: The masked language model, its tokenizer, and the torch device.
      nlp: A spaCy language model for NER.

    Returns:
      float: The computed Identity → Target score.
    """
    # Expand the target words list.
    global TARGETS_ARE_EXPANDED
    if not TARGETS_ARE_EXPANDED:
      set_targets(target_base_words)
    # Mask the target-related words.
    masked_text = mask_entities(text, base_words=TARGETS, nlp=nlp,
                                  entity_labels=["ORG", "NORP", "GPE"])
    # Compute score using the immutable identity words.
    return compute_score(masked_text, IDENTITY_WORDS, model, tokenizer, device, alpha=alpha)

def compute_target_to_identity_score(text, target_base_words, model, tokenizer, nlp, device, alpha=0.5):
    """
    Computes the Target → Identity score.
    
    This function uses the immutable identity words (DEFAULT_IDENTITY_WORDS) to mask
    identity-related words in the text and then computes the log-average probability that
    these masks are filled by target-related candidate words (using the expanded TARGETS list).

    Parameters:
      text (str): The input text.
      target_base_words (set): List of base target-related words.
      model, tokenizer, device: The masked language model, its tokenizer, and the torch device.
      nlp: A spaCy language model for NER.
    
    Returns:
      float: The computed Target → Identity score.
    """
    global TARGETS
    global TARGETS_ARE_EXPANDED
    # Expand the target words list.
    if not TARGETS_ARE_EXPANDED:
      set_targets(target_base_words)
    # Mask the identity-related words.
    masked_text = mask_entities(text, base_words=IDENTITY_WORDS, nlp=nlp)
    # Compute score using the expanded target words.
    return compute_score(masked_text, TARGETS, model, tokenizer, device, alpha=alpha)


def compute_fusion_proximity_score(identity_to_target_score, target_to_identity_score):
    """
    Computes the overall Identity Fusion Score as a harmonic mean of the two directional scores.

    Equation:
        f_{(I,T)} = (2 * S_{I→T} * S_{T→I}) / (S_{I→T} + S_{T→I})

    Parameters:
      identity_to_target_score (float): The score from identity to target.
      target_to_identity_score (float): The score from target to identity.

    Returns:
      float: The harmonic mean fusion score.
    """
    if identity_to_target_score + target_to_identity_score == 0:
        return 0.0
    return (2 * identity_to_target_score * target_to_identity_score) / (identity_to_target_score + target_to_identity_score)
  
  
def compute_kin_fusion_score(text, target_base_words, model, tokenizer, nlp, device, alpha=0.5):
    """
    Computes the Kin → Target score.
    
    This function expands target_base_words and kin vocabularies (via set_targets with gensim),
    masks target-related words in the text (using spaCy NER and the expanded word sets),
    and then computes the log-average probability that these masks are filled by kin-related
    candidate words.

    Parameters:
        text (str): The input text.
        target_base_words (set): List of base target-related words.
        model, tokenizer, device: The masked language model, its tokenizer, and the torch device.
        nlp: A spaCy language model for NER.

    Returns:
        float: The computed Kin → Target score.
    """
    # Expand the target words list.
    global KIN_IF_ARE_EXPANDED, KIN_IF_WORDS, TARGETS, TARGETS_ARE_EXPANDED
    if not KIN_IF_ARE_EXPANDED:
        set_kin_if_words()
    if not TARGETS_ARE_EXPANDED:
        set_targets(target_base_words)
    # Mask the target-related words.
    masked_text = mask_entities(text, base_words=TARGETS, nlp=nlp,
                                entity_labels=["ORG", "NORP", "GPE"])
    # Compute score using the immutable identity words.
    return compute_score(masked_text, KIN_IF_WORDS, model, tokenizer, device, alpha=alpha)
