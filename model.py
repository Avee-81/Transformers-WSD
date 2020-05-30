import xml.etree.ElementTree as ET
import torch
import pickle
import glob
import numpy as np
from tqdm import tqdm
import csv

from transformers import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


class Trans:
    
    def __init__(self, MODEL_CLASS, TOKENIZER_CLASS, WEIGHT_CLASS):
        
        self.tokenizer = TOKENIZER_CLASS.from_pretrained(WEIGHT_CLASS)
        
        self.model = MODEL_CLASS.from_pretrained(WEIGHT_CLASS)
        self.model.eval()
        self.model.cuda()

class Word_Sense_Model:
    
    def __init__(self, MODEL_CLASS, TOKENIZER_CLASS, WEIGHT_CLASS):
        
        self.sense_number_map = {'N':1, 'V':2, 'J':3, 'R':4}
        
        self.TransModel = Trans(MODEL_CLASS, TOKENIZER_CLASS, WEIGHT_CLASS)


        
    def open_xml_file(self, file_name):
        
        tree = ET.parse(file_name)
        root = tree.getroot()
        
        return root, tree
    
    def apply_tokenizer(self, word):
        
        return self.TransModel.tokenizer.tokenize(word)  
        
    def get_embeddings(self, sentence):
        
        input_ids = torch.tensor([self.TransModel.tokenizer.encode(sentence, add_special_tokens=True)])
        input_ids = input_ids.cuda()
        
        with torch.no_grad():
            last_hidden_states = self.TransModel.model(input_ids)

        return last_hidden_states[0][0].cpu().numpy()
        
    def create_word_sense_maps(self, _word_sense_emb):
    
        _sense_emb = {}
        _sentence_maps = {}
        _sense_word_map ={}
        _word_sense_map ={}
    
        for i in _word_sense_emb:
    
            if i not in _word_sense_map:
                _word_sense_map[i] = []

            for j in _word_sense_emb[i]:

                if j not in _sense_word_map:
                    _sense_word_map[j] = []

                _sense_word_map[j].append(i)
                _word_sense_map[i].append(j)

                if j not in _sense_emb:
                    _sense_emb[j] =[]
                    _sentence_maps[j] = []

                _sense_emb[j].extend(_word_sense_emb[i][j]['embs'])
                _sentence_maps[j].extend(_word_sense_emb[i][j]['sentences'])
        
        return _sense_emb, _sentence_maps, _sense_word_map, _word_sense_map
    
    def semeval_sent_sense_collect(self, xml_struct):
        
        _sent =[]
        _sent1 = ""
        _senses = []
        pos = []
        
        for idx,j in enumerate(xml_struct.iter('word')):
            
            _temp_dict = j.attrib
            
            if 'lemma' in _temp_dict:
                
                words = _temp_dict['lemma'].lower()

            else:
                
                words = _temp_dict['surface_form'].lower()
            
            if '*' not in words:
                
                _sent1 += words + " "

                _sent.extend([words])
                
                if 'pos' in _temp_dict:
                    pos.extend([_temp_dict['pos']]*len([words]))

                else:
                    pos.extend([0]*len([words]))
                    
                if 'wn30_key' in _temp_dict:

                    _senses.extend([_temp_dict['wn30_key']]*len([words]))

                else:
                    _senses.extend([0]*len([words]))
                
        return _sent, _sent1, _senses, pos

    def train(self, train_file):
        
        print("Training Embeddings!!")
        
        _word_sense_emb = {}
        
        _train_root, _train_tree = self.open_xml_file(train_file)
        
        for i in tqdm(_train_root.iter('sentence')):
            
            sent, sent1, senses, _ = self.semeval_sent_sense_collect(i)
             
            try:

                    final_layer = self.get_embeddings(sent1)

                    count = 0

                    for idx, j in enumerate(zip(senses, sent)):

                        sense = j[0]
                        word = j[1]

                        if sense != 0:

                            embedding = np.mean(final_layer[count: count+len(self.apply_tokenizer(word)) ],0)

                            if word not in _word_sense_emb:
                                _word_sense_emb[word]={}

                            for s in sense.split(';'):

                                if s not in _word_sense_emb[word]:
                                    _word_sense_emb[word][s]={}
                                    _word_sense_emb[word][s]['embs'] = []
                                    _word_sense_emb[word][s]['sentences'] = []

                                _word_sense_emb[word][s]['embs'].append(embedding)
                                _word_sense_emb[word][s]['sentences'].append(sent1)

                        count += len(self.apply_tokenizer(word)) 

            except Exception as e:
                    print(e)
        
        return _word_sense_emb
   
    def load_embeddings(self, pickle_file_name, train_file):
        
        try:
             
            with open(pickle_file_name, 'rb') as h:
                _x = pickle.load(h)
                
                print("EMBEDDINGS FOUND!")
                return _x
            
        except:
            
            print("Embedding File Not Found!! \n")
            
            word_sense_emb = self.train(train_file)
            
            with open(pickle_file_name, 'wb') as h:
                pickle.dump(word_sense_emb, h)
                
            print("Embeddings Saved to " + pickle_file_name)
            
            return word_sense_emb
        
    def test(self, 
             train_file, 
             test_file, 
             emb_pickle_file,
             save_to, 
             k=1, 
             use_euclidean = False,
             reduced_search = True):
        
        
        word_sense_emb = self.load_embeddings(emb_pickle_file, train_file)

        print("Testing!")
        sense_emb, sentence_maps, sense_word_map, word_sense_map = self.create_word_sense_maps(word_sense_emb)
        
        _test_root, _test_tree = self.open_xml_file(test_file)
        
        _correct, _wrong= [], []
            
        open(save_to, "w").close()
        
        for i in tqdm(_test_root.iter('sentence')):
            

            sent, sent1, senses, pos = self.semeval_sent_sense_collect(i)
            
            final_layer = self.get_embeddings(sent1)

            count, tag, nn_sentences = 0, [], []
            for idx, j in enumerate(zip(senses, sent, pos)):
                
                word = j[1]
                pos_tag = j[2][0]
                
                if j[0] != 0:
                    
                    _temp_tag = 0
                    max_score = -99
                    nearest_sent = 'NONE'

                    embedding = np.mean(final_layer[count:count+len(self.apply_tokenizer(word))],0)
                    
                    min_span = 10000

                    if word in word_sense_map:
                        concat_senses = []
                        concat_sentences = []
                        index_maps = {}
                        _reduced_sense_map = []
                        
                        if reduced_search:
                            
                            for sense_id in word_sense_map[word]:

                                if self.sense_number_map[pos_tag] == int(sense_id.split('%')[1][0]):

                                    _reduced_sense_map.append(sense_id)
                        
                        if len(_reduced_sense_map) == 0 :
                            _reduced_sense_map = list(word_sense_map[word])
                        
                        for sense_id in _reduced_sense_map:
                            index_maps[sense_id] = {}
                            index_maps[sense_id]['start'] = len(concat_senses)

                            concat_senses.extend(sense_emb[sense_id])
                            concat_sentences.extend(sentence_maps[sense_id])

                            index_maps[sense_id]['end'] = len(concat_senses) - 1
                            index_maps[sense_id]['count'] = 0

                            if min_span > (index_maps[sense_id]['end']-index_maps[sense_id]['start']+1):

                                min_span = (index_maps[sense_id]['end']-index_maps[sense_id]['start']+1)

                        min_nearest = min(min_span, k)

                        concat_senses = np.array(concat_senses)

                        if use_euclidean:

                            simis = euclidean_distances(embedding.reshape(1,-1), concat_senses)[0]
                            nearest_indexes = simis.argsort()[:min_nearest]

                        else:
                            simis = cosine_similarity(embedding.reshape(1,-1), concat_senses)[0]
                            nearest_indexes = simis.argsort()[-min_nearest:][::-1]

                        for idx1 in nearest_indexes:

                            for sense_id in _reduced_sense_map:

                                if index_maps[sense_id]['start']<= idx1 and index_maps[sense_id]['end']>=idx1:
                                    index_maps[sense_id]['count'] += 1

                                    score = index_maps[sense_id]['count']

                                    if score > max_score:
                                        max_score = score
                                        _temp_tag = sense_id
                                        nearest_sent = concat_sentences[idx1]


                    tag.append(_temp_tag)
                    nn_sentences.append(nearest_sent)

                count += len(self.apply_tokenizer(word))
                
            _counter = 0
            
            for j in i.iter('word'):
                
                temp_dict = j.attrib
                
                try:
                    
                    if 'wn30_key' in temp_dict:
                        
                        if tag[_counter] == 0:
                            pass
                        
                        else:
                            j.attrib['WSD'] = str(tag[_counter])
                            
                            if j.attrib['WSD'] in str(temp_dict['wn30_key']).split(';') :
                           
                                _correct.append([temp_dict['wn30_key'], j.attrib['WSD'], (sent1), nn_sentences[_counter]])
                            else:
                                _wrong.append([temp_dict['wn30_key'], j.attrib['WSD'], (sent1), nn_sentences[_counter]])

                        _counter += 1
                        
                except Exception as e:
                    
                    print(e)
            
        with open(save_to, "w") as f:
        
            _test_tree.write(f, encoding="unicode")    
        
        print("OUTPUT STORED TO FILE: " + str(save_to))
        
        return _correct, _wrong

MODELS = [
          (AlbertModel,AlbertTokenizer,'albert-xxlarge-v2', 'ALBERT'),
          (BertModel,BertTokenizer,'bert-large-uncased', 'BERT'),
          (CTRLModel,CTRLTokenizer,'ctrl','CTRL'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased','DistilBERT'),
          (ElectraModel,ElectraTokenizer,'google/electra-large-discriminator','ELECTRA'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt','OpenAIGPT'),
          (GPT2Model,GPT2Tokenizer,'gpt2-large', 'OpenAIGPT2'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103', 'TransformerXL'),
          (XLNetModel,XLNetTokenizer,'xlnet-large-cased', 'XLNet')
         ]

for model_class, tokenizer_class, pretrained_weights, model_name in MODELS:

     print("Running " + model_name)
     WSD = Word_Sense_Model(model_class, tokenizer_class, pretrained_weights)
     
     ## SensEval 3 Training and Testing

     train_file = "./data/senseval3task6_train.xml"
     test_file = "./data/senseval3task6_test.xml"
     emb_pickle_file = "./pickles/"+ str(model_name) + "/SE3.pickle"
     save_to = "./results/"+ str(model_name) + "/XML_SE3/"
     save_csv_to ="./results/"+ str(model_name) + "/CSV_SE3/"

     for nn in [1,3,5,7,10]:

        correct, wrong = WSD.test( train_file = train_file, 
                                  test_file = test_file, 
                                  emb_pickle_file = emb_pickle_file,
                                  save_to = save_to + str(nn) + "NN_SE3.xml", 
                                  k=nn, 
                                  use_euclidean = False,
                                  reduced_search = False)
        
        with open(save_csv_to + "Correct_"+ str(nn) + "NN_SE3.csv" , "w") as f:
            writer = csv.writer(f)
            writer.writerows(correct)

        with open(save_csv_to + "Wrong_"+ str(nn) + "NN_SE3.csv" , "w") as f:
            writer = csv.writer(f)
            writer.writerows(wrong)

     ## SensEval 2 Training and Testing

     train_file = "./data/senseval2_lexical_sample_train.xml"
     test_file = "./data/senseval2_lexical_sample_test.xml"
     emb_pickle_file = "./pickles/"+ str(model_name) + "/SE2.pickle"
     save_to = "./results/"+ str(model_name) + "/XML_SE2/"
     save_csv_to ="./results/"+ str(model_name) + "/CSV_SE2/"

     for nn in [1,3,5,7,10]:

        correct, wrong = WSD.test(train_file = train_file, 
                                  test_file = test_file, 
                                  emb_pickle_file = emb_pickle_file,
                                  save_to = save_to + str(nn) + "NN_SE2.xml", 
                                  k=nn, 
                                  use_euclidean = False,
                                  reduced_search = False )
        
        with open(save_csv_to + "Correct_"+ str(nn) + "NN_SE2.csv" , "w") as f:
            writer = csv.writer(f)
            writer.writerows(correct)

        with open(save_csv_to + "Wrong_"+ str(nn) + "NN_SE2.csv" , "w") as f:
            writer = csv.writer(f)
            writer.writerows(wrong)
