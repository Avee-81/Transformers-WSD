import xml.etree.ElementTree as ET
import numpy as np
import glob
import csv

MODELS = [
          'BERT', 'CTRL', 'DistilBERT',
          'OpenAIGPT2','OpenAIGPT', 'TransformerXL',
          'ELECTRA', 'ALBERT', 'XLNet',
          ]

###############################################################################
## Data Stats
def get_stats(file_name):

    tree = ET.parse(file_name)
    root = tree.getroot()

    sentence = 0
    senses = {}
    pos_inf = {}
    words = {}
    total_senses = 0

    for i in root.iter("sentence"):

      sentence += 1

      for j in i.iter('word'):

          dict1 = j.attrib

          if 'wn30_key' in dict1:

              pos_inf[dict1['pos'][0]] = pos_inf.get(dict1['pos'][0], 0) + 1

              for i in dict1['wn30_key'].split(';'):
                
                senses[i] = 1
                total_senses += 1
              
              words[dict1['surface_form'].lower()] = 1

    return [sentence, senses, pos_inf, words, total_senses]

result = [['Data', 'Sentences', 'Senses','Nouns','ADJ','Verbs', 'Words', 'Total Senses']]

files = ['./data/senseval3task6_train.xml',
         './data/senseval3task6_test.xml',
         './data/senseval2_lexical_sample_train.xml',
         './data/senseval2_lexical_sample_test.xml'
        ]

for i in files:

    a = get_stats(i)
    temp = []

    temp.append(i.split('/')[-1][:-4])

    print(temp[0])

    if len(a[2])==3:

      temp.extend([a[0],len(a[1]), a[2]['N'], a[2]['J'], a[2]['V'], len(a[3]), a[4]])

    else:
      print(a[2])

    result.append(temp)

with open('./results/analysis/Data_Stats.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(result)


#######################################################################################
## Error Analysis

result = [['Model_Name','Data', 'Total NN', 'NN', 'Total VB', 'VB', 'Total ADJ', 'ADJ']]

for model in MODELS:

  print(model)

  file_path1 = './results/' + model + '/XLM_SE2/'
  file_path2 = './results/' + model + '/XLM_SE3/'

  ALL_FILES = glob.glob(file_path1 + '*')
  ALL_FILES.extend(glob.glob(file_path2 + '*'))
  # print(ALL_FILES)

  for f in ALL_FILES:
    
    if f.endswith('.xml'):

      reduced_f = f.split('/')[-1][:-4].split('_') ##  [kNN, SE2/SE3]

      reduced_f[0] = int(reduced_f[0][:-2])  ##  [k, SE2/SE3]

      
      if reduced_f[0] == 1:
        
        tree = ET.parse(f)
        root = tree.getroot()

        answer = {}

        answer['N'] = {'Total':0, 'Correct':0}
        answer['V'] = {'Total':0, 'Correct':0}
        answer['J'] = {'Total':0, 'Correct':0}


        for i in root.iter('word'):

            dict1 = i.attrib

            if 'wn30_key' in dict1:

                true_senses = dict1['wn30_key'].split(';')

                answer[dict1['pos'][0]]['Total'] += 1

                
                if 'WSD' in dict1 and dict1['WSD'] in true_senses:

                    answer[dict1['pos'][0]]['Correct'] += 1
                

        if reduced_f[1] == 'SE2':

          temp = [model, 'SE2', answer['N']['Total'], answer['N']['Correct'], answer['V']['Total'], answer['V']['Correct'],
                  answer['J']['Total'], answer['J']['Correct']]

          result.append(temp)

        else:
          
          temp = [model, 'SE3', answer['N']['Total'], answer['N']['Correct'], answer['V']['Total'], answer['V']['Correct'],
                  answer['J']['Total'], answer['J']['Correct']]

          result.append(temp)


with open('./results/analysis/Classification_stats.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(result)

