from transformers import BertModel
import torch.nn as nn
import torch 

class Doc_representation(nn.Module):

	def __init__(self,  word_emb_dim, n_func , device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.W_a = nn.Linear(self.word_emb_dim , 1) # query matrix
		self.W_v = nn.Linear(self.word_emb_dim , self.word_emb_dim) # value matrix
		self.doc_linear = nn.ModuleList([nn.Linear(self.word_emb_dim , 1) for i in range(n_func)])

	def forward(self, func_embed):
		#func_emb (N_func , word dim) 
		#using one attention layer to compute the document embedding
		func_att = self.W_q(func_embed) #shape (N func , 1)
		func_val = self.W_v(func_embed) #shape (N func , word dim)

		attention_matrix = torch.nn.functional.softmax(func_att , dim = 0) #shape (N_func , 1 )
		attention_matrix = torch.transpose(attention_matrix , 0 , 1) #shape (1 , N func )
		result = torch.matmul(attention_matrix ,  func_val) #shape (1 , word_dim)
		result = torch.nn.functional.relu(result.squeeze(0)) # (word dim)

		return result 
	
	def compute_doc_logit(self, doc_emb):
		#doc emb (word)
		doc_logit = []
		for i in range(len(self.doc_linear)):
			doc_logit.append(self.doc_linear[i](doc_emb))
		return doc_logit #list of n_func element
	


