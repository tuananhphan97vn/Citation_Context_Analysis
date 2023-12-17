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

	def forward(self, chunks_idx):
		#chunk idx: list of tokens'index in the citation context
		#--> need to add the cls token and sep token into start/end position of original sequence 
		input = [ [101] + t + [102]  for t in chunks_idx] 
		input = torch.tensor(input)
		last_hidden_state = self.pretrained_model(input) #shape (bs , seq_len , hidden size ) 

		return start_logit , end_logit  

	def get_spare_embed(self , sent_hidden_state , last_hidden_state , batch_bound_sents):
		#this function is used to remove padding vector from tensor. 
		#sent hidden state: (bs , max sent , sent dim ) now will be converted into form of list of tensor (N_sent , sent dim). In this, N_sent can be different between elements 
		#last hidden state: (bs , seq length , word dim) must be converted to form: list of list of element. len of first list is batch size 
		# Second list is list of sentences in each passages. Each list can have different length 
		# Each element in list is tensor. This tensor has shape (N_words , word dim ). N_words is number of word appeared in particular sentence
		# To create the aforementioned object, we use batch bound sent  to extract number of words in sentence as well as number of sentences in passage 
		#batch bound sent is list , each element in list is 2-element list. It express start token and end token in certain sentence
		sent_result , word_result = [] , [] 
		for i in range(len(batch_bound_sents)):
			sent_result.append(sent_hidden_state[i][:len(batch_bound_sents[i])]) #only acquire real sentence 
			list_word_embed = [] 
			for j in range(len(batch_bound_sents[i])):
				start , end = batch_bound_sents[i][j][0] , batch_bound_sents[i][j][1]
				list_word_embed.append(last_hidden_state[i , start : end + 1 , : ]) #shape (end - start , word dim )
			#len list word embed = number of sentence in each passage 
			word_result.append(list_word_embed) # len word result = number of passage in batch = batch size 
		return sent_result , word_result

	def recover_original_sequence(self , last_hidden_state , paragraph_embed_start, paragraph_embed_end , batch_bound_sents):
		start_logit , end_logit = [] , [] 
		for i in range(len(paragraph_embed_start)):
			start_passage , end_passage = batch_bound_sents[i][0][0] , batch_bound_sents[i][-1][1]
			start_logit.append(torch.cat( (last_hidden_state[i , : start_passage , : ] , paragraph_embed_start[i], last_hidden_state[i , end_passage + 1 : , :] ) ))
			end_logit.append(torch.cat( (last_hidden_state[i , : start_passage , : ] , paragraph_embed_end[i], last_hidden_state[i , end_passage + 1  :, :] ) ))
		return torch.stack(start_logit , dim  = 0) , torch.stack(end_logit , dim = 0) #shape (bs , seq length , word dim )

	def forward_pretrained(self, features , max_batch = 8 ):
		all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		output = []
		for i in range(all_input_ids.shape[0]):
			input_id= all_input_ids[i, : ].unsqueeze(0)
			t = self.pretrained_model(input_id)[0] #shape (1 , N_word , word dim)
			output.append(t.squeeze(0)) #shape (N_word , word dim )
		output = torch.stack(output ) #shape (N_chunk , N_word , word dim)
		return output 

	def pool_embed(self, roberta_embed, features , method_pool = 'mean'):
	#roberta embed of each sample in a batch: shape (sequen_length , hidden size)
		results = [] #results is list, one element in result correspond with triple value (question embed , sents_embed, unique_subword_embed) of single feature
		#features can be understood as batch of features
		batch_bound_sents = []
		ques_embed , sents_embed = [] , []
		for i, feature in enumerate(features):
			bound_question, bound_sents  = find_boudaries_single_feature(feature)
			batch_bound_sents.append(bound_sents)
			
			question_embed = pool_sequential_embed(roberta_embed[i] , bound_question[0] , bound_question[1] , method_pool) #shape (hidden_size)
			question_embed = torch.tanh(self.linear_pool(question_embed)) #shape (bs , sent dim )
			#sent emb has different dim with word emb
			batch_sent_embed = [] 
			for bound_sent in bound_sents:
				batch_sent_embed.append(torch.tanh(self.linear_pool(pool_sequential_embed(roberta_embed[i] , bound_sent[0] , bound_sent[1] , method_pool) )))
			batch_sent_embed = torch.stack(batch_sent_embed , axis = 0) #shape (num_sent , hidden_size)	
			
			sents_embed.append(batch_sent_embed)
			ques_embed.append(question_embed)
		ques_embed = torch.stack(ques_embed) #shape (bs , sent dim )
		return ques_embed , sents_embed , batch_bound_sents
