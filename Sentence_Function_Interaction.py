from transformers import BertModel
import torch.nn as nn
import torch 

class Sentence_Function_Interaction(nn.Module):

	def __init__(self,  word_emb_dim, sent_emb_dim, n_iter , device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.n_iter = n_iter
		self.linear_pool = nn.Linear(self.word_emb_dim , self.sent_emb_dim , bias=True)
		self.sent2func = CrossAttentionBlock(self.word_emb_dim)
		self.func2sent = CrossAttentionBlock(self.word_emb_dim)

	def forward(self, sents_emb , func_embed):
		#sents_emb (N_sent , word dim) 
		#func emb (N_func, word dim)
		sents_hid  = sents_emb 
		func_hid = func_embed
		for i in range(self.n_iter):
			#use N iter for getting the hidden state of sentence 
			sents_hid  = torch.nn.functional.relu(sents_hid + self.func2sent(func_hid , sents_hid)) # N sent , word dim 
			func_hid = torch.nn.functional.relu(func_hid + self.sent2func(sents_hid , func_hid)) # N func , word dim
		return sents_hid , func_hid 

class CrossAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, func_embed , sent_embed):

		sent_query = self.W_q(sent_embed) #shape (N_sent , sent dim)
		func_key = self.W_k(func_embed) #shape (N func, word dim)
		func_val = self.W_v(func_embed) #shape (N func , word dim)

		z = 1. / math.sqrt(self.sent_dim) * ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )

		return torch.nn.functional.relu(result) #shape (N_sent, sent dim)

