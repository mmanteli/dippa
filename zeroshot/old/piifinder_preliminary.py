import transformers
import datasets
import torch
import random
import copy
import itertools
import numpy as np
import string

class PiiFinder:

    def __init__(self, model, tokenizer, threshold, use_context=False, choose_n=100, choose_k=10, tokenizer_type="BPE"):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.use_context = bool(use_context)
        self.choose_n = int(choose_n)
        self.choose_k = int(choose_k)
        self.special_tokens = tokenizer.all_special_tokens
        assert tokenizer_type in ["BPE", "WordPiece"]
        self.tokenizer_type = tokenizer_type
        self.continuation_marker = {"BPE": "▁", "WordPiece": "##"}[tokenizer_type]

    def find_pii(self, text, debug = False):
        masked_indices, tokenized_text, decoded_text, context = self.mask(text)
        if debug: print(masked_indices)
        to_be_redacted = []
        to_be_redacted_words = []
        to_redact_with = []
        for ind, cont in zip(masked_indices, context):
            final_score, predictions = self.get_scores(ind, tokenized_text, cont, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            if final_score < self.threshold:
                to_be_redacted.append(ind)
                to_redact_with.append(predictions)
                to_be_redacted_words.append(self.tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"][0][ind]))
                
        return {"decoded_text": decoded_text, "tokenizer_output": tokenized_text, "to_redact_indices": to_be_redacted, "to_redact_words": to_be_redacted_words, "predictions": to_redact_with}

    def print_pii(self, text, debug=False):
        masked_indices, tokenized_text, decoded_text, context = self.mask(text)
        if context == []:
            context = [[]*len(masked_indices)]
        prints = []
        for ind, cont in zip(masked_indices, context):
            final_score, predictions = self.get_scores(ind, tokenized_text, cont, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            if final_score < self.threshold:
                prints.append(f'{word} \t >> {final_score} \t >> Redact')#, preds: {predictions[:4]}')
            else:
                prints.append(f'{word} \t >> {final_score}')
        for p in prints:
            print(p)

#---------------------------------MASKING---------------------------------------#
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
                if converted[i] not in self.special_tokens and indices!=[]:   
                    # here we are only skipping the fact that first token is a special token; indices is empty.
                    indices[-1].append(i)
        return indices

    def mask(self, text):
        if self.use_context:
            return self.context_aware_mask(text)
        t = self.tokenizer(text, return_tensors='pt') # prepare normal tokenized input
        if self.tokenizer_type == "BPE":
            indices = self.get_indices_BPE(t)
        elif self.tokenizer_type == "WordPiece":
            indices = self.get_indices_WordPiece(t)
        return indices, t, self.tokenizer.decode(t.input_ids[0]), [[]]*len(indices)   # []s for empty context


#----------------------------CONTEXT AWARE MASKING---------------------------------#
    def find_same_tokens(self, lst, index):
        target_value = lst[index]
        return [i for i, value in enumerate(lst) if value == target_value and i != index]
    
    def get_context_indices_WordPiece(self, t):
        converted = self.tokenizer.convert_ids_to_tokens(t["input_ids"][0])
        indices=[]
        words = []

        # first, getting the indices as above, but also saving the words in lowercase
        for i in range(0, len(t["input_ids"][0])):
            if converted[i][:2] != self.continuation_marker and converted[i] not in self.special_tokens:
                indices.append([i])
                words.append(converted[i].lower())
            else:
                if converted[i] not in self.special_tokens and indices!=[]:   
                    # here we are only skipping the fact that first token is a special token; indices is empty.
                    indices[-1].append(i)
                    words[-1] += converted[i][2:].lower()

        indices_context=[]
        assert len(words)==len(indices), "Issues with masking the sentence."

        # then, map same words to same context: eg. "Ville" in indices [2,3] and [13,14]
        # => context for first is [13,14] and [2,3] for second.
        for i in range(len(words)):
            ind_of_words = self.find_same_tokens(words, i)
            if ind_of_words != []:
                #print(words[i],":", ind_of_words, np.array(words)[ind_of_words])
                current = []
                for j in ind_of_words:
                    current+= indices[j]
                indices_context.append(current)
            else:
                indices_context.append([])
        
        assert len(indices)==len(indices_context), "Issues with context masking, "+str(len(indices))+"!="+str(len(indices_context))+"\nIndices:\t"+str(indices)+"\nContext:\t"+str(indices_context)
        return indices, indices_context 

    def get_context_indices_BPE(self, t):
        converted = self.tokenizer.convert_ids_to_tokens(t["input_ids"][0])
        indices=[]
        words = []

        reminder = False   # for BPE, to separate punct, we need to know if the last token was punct
        for i in range(0, len(t["input_ids"][0])):
            # for BPE, continuation marker is actually a "starting marker"
            if reminder or (converted[i][0] == self.continuation_marker and converted[i] not in self.special_tokens) or converted[i] in string.punctuation:
                reminder=False
                if converted[i] in string.punctuation:
                    reminder = True
                indices.append([i])
                if converted[i][0] == self.continuation_marker:
                    words.append(converted[i][1:].lower())
                else:
                    words.append(converted[i].lower())
            else:
                if converted[i] not in self.special_tokens and indices!=[]:   
                    # here we are only skipping the fact that first token is a special token; indices is empty.
                    indices[-1].append(i)
                    words[-1] += converted[i].lower()

        indices_context=[]
        assert len(words)==len(indices), "Issues with masking the sentence."

        # then, map same words to same context: eg. "Ville" in indices [2,3] and [13,14]
        # => context for first is [13,14] and [2,3] for second.
        for i in range(len(words)):
            ind_of_words = self.find_same_tokens(words, i)
            if ind_of_words != []:
                #print(words[i],":", ind_of_words, np.array(words)[ind_of_words])
                current = []
                for j in ind_of_words:
                    current+= indices[j]
                indices_context.append(current)
            else:
                indices_context.append([])
        
        assert len(indices)==len(indices_context), "Issues with context masking, "+str(len(indices))+"!="+str(len(indices_context))+"\nIndices:\t"+str(indices)+"\nContext:\t"+str(indices_context)
        return indices, indices_context 

    
    def context_aware_mask(self, text):
        
        t = self.tokenizer(text, return_tensors='pt') # prepare normal tokenized input
        if self.tokenizer_type == "BPE":
            indices, context = self.get_context_indices_BPE(t)
        elif self.tokenizer_type == "WordPiece":
            indices,  context = self.get_context_indices_WordPiece(t)

        return indices, t, self.tokenizer.decode(t.input_ids[0]), context


#-------------------------------predictions-------------------------------------#

    def predict(self, masked, i, true_token, print_results=False):

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

        top_logits, top_tokens = torch.sort(logits, dim=2, descending=True)#[:,:,:self.choose_n]
        top_tokens = top_tokens[:,:,:self.choose_n]
        top_guesses = [self.tokenizer.decode(g) for g in top_tokens[0,i,:]]
        # Do only in debug mode:
        if print_results:
            print(f'{self.tokenizer.decode(true_token)} has probability {word_probability}')
            # see choose_n predictions for debug
            top_probs = to_probability(top_logits[0,i,:])
            top_logits = top_logits[:,:,:self.choose_n]
            top_tokens = top_tokens[:,:,:self.choose_n]
    
        
            print("Guesses:",self.tokenizer.decode(top_tokens[0,i,:]))
            #print("Logits: ",top_logits[0,i,:])
            #print("Probs:  ",top_probs[:self.choose_n])
            print("")
        return word_probability, top_guesses

    def get_scores(self, to_be_masked, tokens, context, debug):
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
        predictions = []
    
        # loop over the subtokens of a word
        for i in range(len(to_be_masked)):
            # making a deep copy as tensors are nested and yada yada
            t = copy.deepcopy(tokens)
            current = to_be_masked[i:]   # this is the token we are CURRENTLY interested in
            for j in current:
                t["input_ids"][0][j] = self.tokenizer.mask_token_id     # we mask the SUBtokens that are in current
            if context != []:   # if we have context, mask that
                for j in context:
                    t["input_ids"][0][j] = self.tokenizer.mask_token_id 
            if debug:
                print(self.tokenizer.decode(t["input_ids"][0]))
            # multiply the final score with the predicted probability => aggregates over to_be_masked==one word
            score, preds = self.predict(t, current[0], tokens.input_ids[0][current[0]], print_results=debug)
            final_score *= score
            predictions.append(preds)
            
        return final_score, predictions

    
