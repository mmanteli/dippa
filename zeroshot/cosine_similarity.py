import numpy as np
import torch
#from transformers import pipeline, pipelines, AutoTokenizer, AutoModelForPreTraining
from sklearn.metrics.pairwise import cosine_similarity

def get_embed_for_a_vector(x, pipeline):
    #p=pipeline(task="feature-extraction",model=MODEL_NAME,return_tensors=True,device=0)
    embedded = pipeline(x)
    embedded_pooled=[torch.mean(elem,axis=1).cpu() for elem in embedded]
    results=torch.vstack(embedded_pooled).numpy()
    return results

def cosine_sim(x,y,choose_k):
    M=cosine_similarity(x,y)
    aligned=np.argsort(-M,axis=-1)

    all_sims = []
    for k in range(1, choose_k+1):
        sims=[]
        for i in range(M.shape[0]): #M.shape[0] is the number of rows / input documents
            j=aligned[i,k] # index [1] for 2nd best match => index [0] gives the same words (the match perfectly).
            score=M[i,j]
            sims.append((i,j,score))
        # sort in descending order  element -> score => sort by score
        sims.sort(key=lambda element:element[2],reverse=True)
        all_sims.append(sims)
    return all_sims

def find_best_sim(sims, word_index):
    # word index == 0 => we are finding matches to the original word
    for sim in sims:
        if sim[0] == word_index:
            return sim
    return False

def find_best_cosine_match(word, preds, choose_k, pipe):
    if word in preds:
        preds.remove(word)
    vector = [word]+preds
    emb=get_embed_for_a_vector(vector, pipe)
    k_sims = cosine_sim(emb,emb, choose_k)
    results = []
    for sims in k_sims:
        winner_sim = find_best_sim(sims, word_index=0)
        assert vector[winner_sim[0]] == word
        results.append(vector[winner_sim[1]])
    # winner_sim[0] == index of word
    # winner_sim[1] == index of most similar word
    # winner_sim[2] == the similarity between the two
    
    return results
    
    