import transformers
import torch
import spacy
import random
import copy
import itertools
import numpy as np
import string
from tqdm import tqdm
import csv


class PiiDetector:

    def __init__(self, model, tokenizer, lemmatizer, threshold, use_context=False, choose_n=10, choose_k=3, embedding_model=None, tokenizer_type=None, return_tokenizer_output=False, continuation_marker=None):
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device}')
        self.model = model.to(self.device)
        self.model.eval()   # remove some unneeded functionality
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        if tokenizer_type is None:
            if isinstance(tokenizer, transformers.BertTokenizer) or isinstance(tokenizer, transformers.BertTokenizerFast) or isinstance(tokenizer, transformers.RobertaTokenizerFast):
                tokenizer_type="WordPiece"
            elif isinstance(tokenizer, transformers.XLMRobertaTokenizer) or isinstance(tokenizer,transformers.XLMRobertaTokenizerFast):
                tokenizer_type="BPE"
        assert tokenizer_type in ["BPE", "WordPiece"], f"Tokenizer type not automatically detected ({type(tokenizer)}). Define as BPE or WordPiece, SentencePiece not implemented, but returns similar values to BPE so it might work."
        self.tokenizer_type = tokenizer_type
        self.continuation_marker = {"BPE": "▁", "WordPiece": "##"}[tokenizer_type] if continuation_marker is None else continuation_marker
        self.special_tokens = tokenizer.all_special_tokens
        self.threshold = threshold    # this is to be optimized
        self.use_context = bool(use_context)
        self.choose_n = int(choose_n)
        self.choose_k = int(choose_k)
        self.return_tokenizer_output=return_tokenizer_output
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = self.model

    def find_pii(self, text, debug = False):
        masked_indices, tokenized_text, decoded_text, context = self.mask(text)
        if context == []:
            context = [[]*len(masked_indices)]
        if debug: print(masked_indices)
        to_be_redacted = []
        to_be_redacted_words = []
        to_redact_with = []
        to_redact_context = []
        for ind, cont in zip(masked_indices, context):
            final_score, predictions = self.get_scores(ind, tokenized_text, cont, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            if final_score < self.threshold:
                to_be_redacted.append(ind)
                to_redact_with.append(predictions)
                to_be_redacted_words.append(self.tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"][0][ind]))
                to_redact_context.append(cont)
        if self.return_tokenizer_output:
            return {"original_text": text,
                    "decoded_text": decoded_text, 
                    "tokenizer_output": tokenized_text, 
                    "to_redact_indices": to_be_redacted, 
                    "to_redact_words": to_be_redacted_words, 
                    #"predictions": to_redact_with,
                    "context": to_redact_context}
        else:
            return {"original_text": text,
                    "decoded_text": decoded_text,
                    "to_redact_indices": to_be_redacted, 
                    "to_redact_words": to_be_redacted_words,
                    #"predictions": to_redact_with,
                    "context": to_redact_context}

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

    def redact_pii(self, text, debug=False):
        masked_indices, tokenized_text, decoded_text, context = self.mask(text)
        if context == []:
            context = [[]*len(masked_indices)]
        prints = []
        for ind, cont in tqdm(zip(masked_indices, context), total=len(masked_indices)):
            final_score, predictions, _ = self.get_scores(ind, tokenized_text, cont, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            if final_score < self.threshold:
                prints.append(f'█████')#, preds: {predictions[:4]}')
            else:
                prints.append(f'{word}')
        for p in prints:
            print(p, end=" ")

    


    def score_pii(self, text, filename, debug=False):
        """
        Function for scoring each word in a text. Facilitates calculation of optimal threshold with
        only one model iteration.
        Inputs:
            PiiDetector object
            text: text to be scored in string format
            filename: filename to which save the results, .csv appended
        Returns:
            None, writes to filename.csv
        """
        def flatten(xss):
            return [x for xs in xss for x in xs]

        def parse_again(m_ind, c_ind):
            new_m = []
            new_c = []
            already_parsed = []
            for m,c in zip(m_ind, c_ind):
                if m not in already_parsed:
                    new_m.append(m)
                    new_c.append(c)
                    already_parsed.append(m)
            return new_m, new_c
        torch.cuda.empty_cache()
        masked_indices, tokenized_text, decoded_text, context = self.mask(text)
        if context == []:
            context = [[]*len(masked_indices)]
        prints = []
        new_prints = []
        #print(masked_indices)
        #print(context)

        if not flatten(masked_indices) == range(masked_indices[0][0], masked_indices[-1][-1]+1):
            masked_indices, context = parse_again(masked_indices, context)
        #for i,j in zip(masked_indices, context):
        #    print(f"{i}, {j}: {self.tokenizer.decode(tokenized_text['input_ids'][0][i])} ")
        
        for ind, cont in tqdm(zip(masked_indices, context), total=len(masked_indices)):
            final_score, predictions, all_scores = self.get_scores(ind, tokenized_text, cont, debug)
            word = self.tokenizer.decode(tokenized_text["input_ids"][0][ind])
            all_words = [self.tokenizer.decode(tt) for tt in tokenized_text["input_ids"][0][ind]]
            prints.append(f'{final_score}\t{word}\n')
            for new_word, new_score in zip(all_words, all_scores):
                new_prints.append(f'{final_score}\t{new_word}\t{new_score}\n')

        #print(prints)
        with open(filename+"_by_word.tsv", "w") as f:
            for row in prints:
                f.write(row)
        with open(filename+"_by_token.tsv", "w") as f:
            for row in new_prints:
                f.write(row)
        


    
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
        t = self.tokenizer(text, return_tensors='pt')#.to(self.device) # prepare normal tokenized input, to GPU later
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
        if self.lemmatizer:
            lem_words = [t.lemma_.lower() for t in self.lemmatizer(" ".join(words))]   # lemmatizer randomly capitalizes
        else:
            lem_words = words
        assert len(words)==len(indices), "Issues with masking the sentence."
        assert len(lem_words)==len(indices), f'Issues with lemmatizing the sentence.\n{lem_words}\n{words}'
    
        # then, map same words to same context: eg. "Ville" in indices [2,3] and [13,14]
        # => context for first is [13,14] and [2,3] for second.
        for i in range(len(lem_words)):
            ind_of_words = self.find_same_tokens(lem_words, i)
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

        to_be_appended = False  # this is flag for continuation of word
        reminder=False  # this is a reminder to separate a punctuation
        for i in range(0, len(t["input_ids"][0])):
            #print("now in converted", converted[i])
            if converted[i] in self.special_tokens:
                continue
            for j in converted[i]:
                if j in self.continuation_marker:
                    to_be_appended=False  # new word
                    continue
                if j in string.punctuation and j != self.continuation_marker:
                    to_be_appended=False
                    reminder=True
                if to_be_appended and not reminder:
                    words[-1]+= j.lower()
                    if i not in indices[-1]:
                        indices[-1].append(i) #nothing for indices, same word
                else:
                    words.append(j.lower())
                    indices.append([i])
                    to_be_appended=True
                if j not in string.punctuation:
                    reminder=False

        indices_context=[]
        if self.lemmatizer:
            lem_words = [t.lemma_.lower() for t in self.lemmatizer(" ".join(words))]   # lemmatizer sometimes randomly capitalizes
        else:
            lem_words=words
        #for t,m in zip(words, lem_words):
        #    print(f"decoded {t}, lemmatized {m}")
        assert len(words)==len(indices), "Issues with masking the sentence."
        assert len(lem_words)==len(indices), f'Issues with lemmatizing the sentence.\n{lem_words}\n{words}'

        # then, map same words to same context: eg. "Ville" in indices [2,3] and [13,14]
        # => context for first is [13,14] and [2,3] for second.
        for i in range(len(words)):
            ind_of_words = self.find_same_tokens(lem_words, i)
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
        
        t = self.tokenizer(text, return_tensors='pt') #.to(self.device) # prepare normal tokenized input, put it to GPU later
        if self.tokenizer_type == "BPE":
            indices, context = self.get_context_indices_BPE(t)
        elif self.tokenizer_type == "WordPiece":
            indices,  context = self.get_context_indices_WordPiece(t)
    
        return indices, t, self.tokenizer.decode(t.input_ids[0]), context


#-------------------------------predictions-------------------------------------#

    def batched_prediction(self, masked_input, debug=False):
        max_length = self.tokenizer.model_max_length
        overlap = int(max_length/2)-1 # to facilitate adding additional CLS etc

        # get these to a more indexed format
        input_ids = masked_input["input_ids"].squeeze(0)
        attention_mask = masked_input["attention_mask"].squeeze(0)
        sanity_check = []

        # this to make overlappig work
        #last_handled_index=0
        batched_result=0
        indices_upper_limits = [i-1+overlap for i in range(overlap, len(input_ids)+overlap, overlap)]
        indices_lower_limits = [i for i in range(0, len(input_ids), overlap)]
    
        if debug:
            print(indices_upper_limits)
            print(indices_lower_limits)
        
        for start, end in zip(indices_lower_limits, indices_upper_limits):
            #print("\nstart and end", start, end)
            #print("shape of input ", input_ids.size())
            if start == 0:  # no need to add bos/cls to beginning
                chunk_input_ids = torch.cat((input_ids[start:end],input_ids[-2:-1]))
                chunk_attention_mask = torch.cat((attention_mask[start:end],torch.tensor([0]).to(self.device)))
            else:
                chunk_input_ids = torch.cat((input_ids[0:1],input_ids[start:end],input_ids[-2:-1]))
                chunk_attention_mask = torch.cat((torch.tensor([0]).to(self.device), attention_mask[start:end],torch.tensor([0]).to(self.device)))
            if debug: print(f'\nlen of inputs {len(chunk_input_ids)}, form index {start} to {end}')
            #if len(chunk_input_ids) < max_length:  # Pad the chunk if it's smaller than max len
            #    padding_length = max_length - len(chunk_input_ids)
            #    chunk_input_ids = torch.cat([chunk_input_ids, torch.zeros(padding_length, dtype=torch.long).to(self.device)])
            chunk_t = {"input_ids": chunk_input_ids.unsqueeze(0), "attention_mask":chunk_attention_mask.unsqueeze(0)}
            #print("chunked decoded ", self.tokenizer.decode(chunk_t["input_ids"][0]))
            #print("chunked last 3 tokens ",chunk_t["input_ids"][-3:])
            with torch.no_grad():
                model_out = self.model(**chunk_t)
            if self.tokenizer_type=="WordPiece":
                logits = model_out["prediction_logits"]
            elif self.tokenizer_type=="BPE":
                logits = model_out["logits"]
            if debug: print(f"output size {logits.size()}")
            #change to cpu: cat is equally fast on cpu and this saves space
            logits_c = logits.detach().cpu()   # detach maybe not necessary since we do torch.no_grad() but oh well
            del logits
            #print("adding result")
            if torch.is_tensor(batched_result):
                batch_logits = logits_c[:,overlap:-1,:] # remove cls etc. bos removal already included in overlap
                #print("selected size ", batch_logits.size())
                if debug: print("Should add selected", batch_logits.size(), "to existing ", batched_result.size() )
                batched_result=torch.cat((batched_result, batch_logits),dim=1)
                sanity_check = torch.cat((sanity_check, chunk_t["input_ids"][:,overlap:-1]), dim=1)
                #print("result was", batched_result.size())
            else:
                batched_result = logits_c[:,:-1,:]
                sanity_check = chunk_t["input_ids"][:,:-1] # select same indices and see if everything matches up
            if debug: print("size of concatenated logits;", batched_result.size())
        assert torch.equal(sanity_check, masked_input["input_ids"]), "Batched prediction resulted in error. This is likely BOS/EOS/CLS token error or indexing problem. This has not been tested for models with odd (as op. even) input length"
        return batched_result.to(self.device)  # back to gpu


    def predict(self, masked, i, true_token, print_results=False, return_predictions=False):

        def to_probability(A):
            softmax = torch.nn.Softmax(dim=0)
            return softmax(A)

        # setting this to device only here to save on memory
        masked.to(self.device)
        # do a prediction
        with torch.no_grad():
            # do overlapping batch if too large input
            if masked.input_ids.size()[1] > self.tokenizer.model_max_length:
                #if self.debug:
                #print(f"Text too long ({masked.input_ids.size()[1]}). Batching.")
                logits = self.batched_prediction(masked)
            else:
                model_out = self.model(**masked)
                if self.tokenizer_type=="WordPiece":
                    logits = model_out["prediction_logits"]
                elif self.tokenizer_type=="BPE":
                    logits = model_out["logits"]
        masked.to("cpu") # putting this bach to cpu to save memory
        #print("LOGITS SIZE",logits.size())
        # logits for this word specifically
        logits_i = logits[0,i,:]  # this contains the probabilities for this token
        # change to probability
        probs = to_probability(logits_i)
        # true token is the index
        word_probability = probs[true_token]

        # Originally, we did this:
        #top_logits, top_tokens = torch.sort(logits, dim=2, descending=True)#[:,:,:self.choose_n]
        #top_tokens = top_tokens[:,:,:self.choose_n]
        # However it takes too much memory, this is far better:
        if return_predictions or print_results:
            top_logits, top_tokens = torch.topk(logits, self.choose_n, dim=2, largest=True)
            top_guesses = [self.tokenizer.decode(g) for g in top_tokens[0,i,:]]
        else:
            top_guesses = []
        #if print_results: print(f'{self.tokenizer.decode(true_token)} has best guesses {top_guess} and probability {word_probability}')
  
        # Do only in debug mode:
        if print_results:
            print(f'{self.tokenizer.decode(true_token)} has probability {word_probability}')
            # see choose_n predictions for debug
            top_probs = to_probability(top_logits[0,i,:])
            top_logits = top_logits[:,:,:self.choose_n]
            top_tokens = top_tokens[:,:,:self.choose_n]
    
        
            #print("Guesses:",self.tokenizer.decode(top_tokens[0,i,:]))
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
        all_scores = []
    
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
            all_scores.append(score)
            predictions.append(preds)
            
        return final_score, predictions, all_scores

    