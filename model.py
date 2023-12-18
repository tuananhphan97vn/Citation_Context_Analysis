from transformers import BertModel
import torch.nn as nn
import torch 

class Model(nn.Module):

	def __init__(self, pretrained_model, word_emb_dim, sent_emb_dim, device):
		super().__init__()
		self.device = device
		self.pretrained_model = pretrained_model
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.sent_emb_dim = sent_emb_dim # this value usually is set by 64 
		self.linear_pool = nn.Linear(self.word_emb_dim , self.sent_emb_dim , bias=True)
		#self.sent_encoder = SentenceEncoder(self.n_block, self.sent_emb_dim, self.device)
#		self.classifier_layer = Classifier(self.sent_emb_dim, self.word_emb_dim, self.device)

	def pool_sent_embed(self , last_hidden_state , features,  batch_index):
		batch_feature = [features[t] for t in batch_index]
		ques_embed , list_sent_emb , batch_bound_sents = self.pool_embed(last_hidden_state , batch_feature) #init embed contain question embedding and list of sentence embedding 
		return ques_embed , list_sent_emb , batch_bound_sents

	def forward(self, chunks_idx , tok2sent_ind, func_embed):
		#chunk idx: list of tokens'index in the citation context
		#--> need to add the cls token and sep token into start/end position of original sequence 
		#func embed shape (N_function , func hidden size)
		input = [ [101] + t + [102]  for t in chunks_idx] 
		last_hidden_state = self.forward_pretrained(input) #shape (bs , seq_len , hidden size ) 
		toks_embed = self.extract_tok_emb(last_hidden_state) #shape (n_tok , word dim)
		sents_embed = self.sent_pool(toks_embed , tok2sent_ind) #shape (N_sent , sent dim ) 
		sents_hid , func_hid = self.sent_func_interaction(sents_embed , func_embed) #(N_sent , word dim) (N_func, word dim)
		doc_emb = self.doc_pool(func_hid) #(word dim)
		doc_logit = self.compute_doc_logit(doc_emb) # (N_func) ---> use for multi binary loss 
		return doc_logit   

	def extract_tok_emb(self, list_hidden_state):
		result = [] 
		for t in list_hidden_state:
			result.append(t[1:-1,:])
		result = torch.cat(result , dim = 0)
		return result 

	def forward_pretrained(self, input ):
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		output = []
		for i in range(len(input)):
			# input_id= all_input_ids[i, : ].unsqueeze(0)
			x  = torch.tensor(input[i] , dtype=torch.long).to(self.device).unsqueeze(0) #shape (1 , N_word)
			t = self.pretrained_model(x)[0] #shape (1 , N_word , word dim)
			output.append(t.squeeze(0)) #shape (N_word , word dim )
		# output = torch.stack(output ) #shape (N_chunk , N_word , word dim)
		return output 

	def sent_pool(self, tok_embed, tok2sent_ind ,  method_pool = 'mean'):
		#tok_embed: shape (sequen_length , hidden size)
		bound_sents = find_boudary_sents(tok2sent_ind)
			
		sents_emb = [] 
		for bound_sent in bound_sents:
			sents_emb.append(torch.tanh(self.linear_pool(pool_sequential_embed(tok_embed , bound_sent[0] , bound_sent[1] , method_pool) )))
		sents_emb = torch.stack(sents_emb , dim = 0) #shape (N_sent , sent dim )

		return sents_emb

def pool_sequential_embed(roberta_embed , start , end , method):
	if method =='mean':
		sub_matrix = roberta_embed[start:end+1 , :] 
		return torch.mean(sub_matrix , axis = 0 ) 

def find_boudary_sents(tok2sent_idx):
	#bound sents
	sent_tok = {}

	for i in range(len(tok2sent_idx)):
		if tok2sent_idx[i] not in sent_tok:
			sent_tok[tok2sent_idx[i]] = [i]
		else:
			sent_tok[tok2sent_idx[i]].append(i)

	bound_sents = [] 
	for sent_id in sent_tok:
		bound_sents.append([sent_tok[sent_id][0],sent_tok[sent_id][-1]])

	return bound_sents