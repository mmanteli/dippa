import transformers
import datasets
import torch
import random
import copy
import itertools
import numpy as np
import string

class PiiFinder:

    def __init__(self, model, tokenizer, threshold, tokenizer_type="BPE"):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.special_tokens = tokenizer.all_special_tokens
        assert tokenizer_type in ["BPE", "WordPiece"]
        self.tokenizer_type = tokenizer_type
        self.continuation_marker = {"BPE": "▁", "WordPiece": "##"}[tokenizer_type]

    def find_pii(self, text, debug = False):
        masked_indices, tokenized_text, decoded_text = self.mask(text)
        if debug: print(masked_indices)
        #result = np.zeros(len(tokenized_text["input_ids"][0]), dtype=int)
        result2 = []
        for ind in masked_indices:
            final_score = self.get_scores(ind, tokenized_text, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            if final_score < self.threshold:
                #result[ind] += 1
                result2.append(ind)
        #return result, result2
        return result2

    def print_pii(self, text, debug=False):
        masked_indices, tokenized_text, decoded_text = self.mask(text)
        prints = []
        for ind in masked_indices:
            final_score = self.get_scores(ind, tokenized_text, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            if final_score < self.threshold:
                prints.append(f'{word} \t >> {final_score} \t >> Redact')
            else:
                prints.append(f'{word} \t >> {final_score}')
        for p in prints:
            print(p)

    
    def get_indices_WordPiece(self, t):
        converted = self.tokenizer.convert_ids_to_tokens(t["input_ids"][0])
        indices=[]
        for i in range(0, len(t["input_ids"][0])):
            if converted[i][:2] != self.continuation_marker and converted[i] not in self.special_tokens:
                indices.append([i])
            else:
                if converted[i] not in self.special_tokens and indices!=[]:   # here we are only skipping the fact that first token is a special token; indices is empty.
                    indices[-1].append(i)
        return indices

    def get_indices_BPE(self, t):
        converted = self.tokenizer.convert_ids_to_tokens(t["input_ids"][0])
        indices=[]
        reminder = False   # for BPE, to separate punct, we need to know if the last token was punct
        for i in range(0, len(t["input_ids"][0])):
            # for BPE, continuation marker is actually a "starting marker"
            if reminder or (converted[i][0] == self.continuation_marker and converted[i] not in self.special_tokens) or converted[i] in string.punctuation:
                reminder=False
                if converted[i] in string.punctuation:
                    reminder = True
                indices.append([i])
            else:
                if converted[i] not in self.special_tokens and indices!=[]:   # here we are only skipping the fact that first token is a special token; indices is empty.
                    indices[-1].append(i)
        return indices

    def mask(self, text):
        t = self.tokenizer(text, return_tensors='pt') # prepare normal tokenized input
        if self.tokenizer_type == "BPE":
            indices = self.get_indices_BPE(t)
        elif self.tokenizer_type == "WordPiece":
            indices = self.get_indices_WordPiece(t)
        return indices, t, self.tokenizer.decode(t.input_ids[0])


    

    def predict(self, masked, i, true_token, print_results=False, top=10):

        def to_probability(A):
            softmax = torch.nn.Softmax(dim=0)
            return softmax(A)
            
        # do a prediction
        model_out = self.model(**masked)
        if self.tokenizer_type=="WordPiece":
            logits = model_out["prediction_logits"]
        elif self.tokenizer_type=="BPE":
            logits = model_out["logits"]
    
        # logits for this word specifically
        logits_i = logits[0,i,:]  # this contains the probabilities for this token
        # change to probability
        probs = to_probability(logits_i)
        # true token is the index
        word_probability = probs[true_token]
    
        # Do only in debug mode:
        if print_results:
            print(f'{self.tokenizer.decode(true_token)} has probability {word_probability}')
            # see 10 top predictions for debug
            top_logits, top_tokens= torch.sort(logits, dim=2, descending=True)#[:,:,:top]
            top_probs = to_probability(top_logits[0,i,:])
            top_logits = top_logits[:,:,:top]
            top_tokens = top_tokens[:,:,:top]
    
        
            print("Guesses:",self.tokenizer.decode(top_tokens[0,i,:]))
            print("Logits: ",top_logits[0,i,:])
            print("Probs:  ",top_probs[:top])
            print("")
        return word_probability

    def get_scores(self, to_be_masked, tokens, debug):
        """
        Calculates the (aggregated) probability of the given word based on the model prediction.
        For multi-subtoken words, aggregation strategy is gradual unmasking and multiplication.
        Input: 
            tokens: tokenizer output for a span of text
            to_be_masked: indices for which are masked from the tokens and over which we calculate
                          i.e. indices of the subtokens that form a word.
            debug (False): prints out extra information if True
        Returns:
            (aggregated) probability \in (0,1)
        """
        # initialize the score; we're multiplying, so 1
        final_score = 1
    
        # loop over the subtokens of a word
        for i in range(len(to_be_masked)):
            # making a deep copy as tensors are nested and yada yada
            t = copy.deepcopy(tokens)
            current = to_be_masked[i:]   # this is the token we are CURRENTLY interested in
            for j in current:
                t["input_ids"][0][j] = self.tokenizer.mask_token_id     # we mask the SUBtokens that are in current
            if debug:
                print(self.tokenizer.decode(t["input_ids"][0]))
            # multiply the final score with the predicted probability => aggregates over to_be_masked==one word
            final_score *= self.predict(t, current[0], tokens.input_ids[0][current[0]], print_results=debug)
            
        return final_score
        
    