import pickle
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib
import pylab

FONT_SIZE = 40
LEGEND_FONT_SIZE = 40
markersize = 250
params = {'legend.fontsize': LEGEND_FONT_SIZE,
          'figure.figsize': (17, 17),
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize':FONT_SIZE,
         'xtick.labelsize':FONT_SIZE,
         'ytick.labelsize':FONT_SIZE,
         'axes.labelweight':'black',
         'font.weight':'black'}
plt.rcParams.update(params)

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]
allcolors = ['violet','indigo','blue','green','teal','orange','red','yellow','turquoise','darkcyan','lightblue']

definition={}
definition['bank%1:17:01::'] = "Sloping Land"
definition['bank%1:14:00::'] = "A Financial \n Institution"
definition['bank%1:14:01::'] = "Arrangment of Objects"
definition['bank%1:17:00::'] = "A Long Ridge"
definition['bank%1:21:00::'] = "A Reserve Supply"
definition['bank%1:21:01::'] = "Gambling Games Fund"
definition['bank%1:17:02::'] = "A Road's turn Slope"
definition['bank%1:06:00::'] = "A Bank Building"
definition['bank%1:04:00::'] = "A Flight Maneuver"

def get_tsne(file_path, words, save_path, model_name):

      with open(file_path ,'rb') as h:
          data=pickle.load(h)
      
      final_bert_tsnes = {}

      for word in words:

          a=[]
          
          final_bert_tsnes[word] = {}
          for i in data[word].keys():
              final_bert_tsnes[word][i] = {}
              final_bert_tsnes[word][i]['start'] = len(a)
              a.extend(data[word][i]['embs'])
              final_bert_tsnes[word][i]['end'] = len(a)
          
          a=np.array(a)
          if model_name == 'GPT2':
            a = np.nan_to_num(a, nan=0, posinf=10, neginf=-10)
          
          X_embedded = TSNE(n_components=2).fit_transform(a)
          final_bert_tsnes[word]['tsne'] = X_embedded

      for word in words:

          fig = plt.figure()
          figLegend = pylab.figure()
          ax = fig.add_subplot(1, 1, 1)
          
          for idx,j in enumerate(final_bert_tsnes[word]):
            
              if j!='tsne':
                
                  start = final_bert_tsnes[word][j]['start']
                  end = final_bert_tsnes[word][j]['end']
                  
                  if end - start > 2:

                      embds = final_bert_tsnes[word]['tsne'][start:end]

                      x,y = embds.T
                      ax.scatter(x,y, c=allcolors[idx], s=markersize, label = str(definition[j] + ":"+str(end-start)))


          handles, labels = ax.get_legend_handles_labels()
          labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: -int(t[0].split(':')[1]) ))
          labels = list(labels)
          ax.legend(handles, labels, loc = 2,prop={'size': 0})
          fig.savefig(save_path + word + '.png')
          pylab.figlegend(*(handles, labels))
          figLegend.savefig(save_path + word + '_legend.png')

models = [
          'BERT', 'CTRL', 'DistilBERT', 'OpenAIGPT2', 
          'OpenAIGPT', 'TransformerXL', 'XLNet','ELECTRA', 'ALBERT',
          ]

for model in models:
    print(model)
    path = './pickles/'+ model + '/SE3.pickle'
    save_path = './results/' + model + '/'

    get_tsne(path, ['bank'], save_path, model)
