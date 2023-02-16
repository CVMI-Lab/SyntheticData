import sys, random
from keytotext import pipeline
import pickle

labels = \
['a Annual Crop Land', 'a Forest', 'a Herbaceous Vegetation Land', 'a Highway or Road', 'a Industrial Building', 'a Pasture Land', 'a Permanent Crop Land', 'a Residential Building', 'a River', 'a Sea or Lake']


nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")

def word2sentence(classnames, num=200, save_path=''):
    sentence_dict = {}
    for n in classnames:
        sentence_dict[n] = []
    for n in classnames:
        for i in range(num+50):
            sentence = nlp([n], num_return_sequences=1, do_sample=True)
            sentence_dict[n].append(sentence)

    # remove duplicate
    sampled_dict = {}
    for k, v in sentence_dict.items():
        v_unique = list(set(v))
        sampled_v = random.sample(v_unique, num)
        sampled_dict[k] = sampled_v

    r = open(save_path,"wb")
    pickle.dump(sampled_dict, r)
    r.close()

if __name__ == "__main__":
    num = sys.argv[1]
    save_path = sys.argv[2]
    word2sentence(labels, int(num), save_path)

'''
python3.7 src/LE.py 200 /path/to/save/dataset.pkl
'''