''' 
take word2vec style parallelograms for:
    - sum of concept head OVs lm_head(OV(Tokyo_ell) - OV(Japan_ell) + OV(Germany_ell)) = Berlin?
    - sum of token head OVs (same thing)
    - sum of all the head OVs (same thing)
    - raw head activations 

do this for all tasks, layers, and concept/token/all/raw

# tasks for word2vec 
task_list = [
    'capital-common-countries', 'capital-world', 'currency',
    'city-in-state', 'family', 
    'gram1-adjective-to-adverb',
    'gram2-opposite', 'gram3-comparative', 'gram4-superlative',
    'gram5-present-participle', 'gram6-nationality-adjective',
    'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs'
]
'''
import os 
import json 
import argparse
from nnsight import LanguageModel
from parallelograms import get_neighbors, calculate_save_scores

def run_rank_scan(this_task, task_name, model, layer, concept_k, token_k, w_prefix, dataset):
    ranks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    sep = ' ' if dataset == 'word2vec' else '\t'
    w_prefixes = (w_prefix, w_prefix) # just always keep left prefix = right prefix. 
    
    # for head_ordering in ['token', 'concept', 'all']:
    for head_ordering in ['all']: # TODO uncomment 
        k = {
            'token' : token_k,
            'concept' : concept_k,
            'all' : None 
        }[head_ordering]

        for rank in ranks:
            print(task_name, w_prefix, this_task[0].split(sep)[0], rank)
            neighbors = get_neighbors(
                this_task, model, layer, head_ordering, k, w_prefixes, dataset, rank=rank
            )
            calculate_save_scores(
                this_task, neighbors, model, k, head_ordering, dataset, task_name, layer, w_prefixes, rank
            )            
            del neighbors

def get_optimal_layers(task_list, dataset, with_prefix=False):
    layers = [0, 4, 8, 12, 16, 20, 24, 28] # exclude 31 
    optimal_layers = {}
    superfolder = 'with_prefix' if with_prefix else 'no_prefix'
    for task in task_list:
        concept_values = []
        token_values = []
        for layer in layers:
            fname = f'layer{layer}_results.json'
            with open(f'../cache/parallelograms/{dataset}/{superfolder}/concept/{task}/{fname}', 'r') as f:
                concept_values.append((layer, json.load(f)['nn_acc']))
            
            with open(f'../cache/parallelograms/{dataset}/{superfolder}/token/{task}/{fname}', 'r') as f:
                token_values.append((layer, json.load(f)['nn_acc']))
        
        concept_max = ('concept',) + max(concept_values, key=lambda t: t[1])
        token_max = ('token',) + max(token_values, key=lambda t: t[1])
        overall = max([concept_max, token_max], key=lambda t: t[-1])
        print(task, overall)
        optimal_layers[task] = overall
    return optimal_layers

def main(args):
    categories = {}
    for task in os.listdir(f'../data/{args.dataset}/'):
        if '.txt' in task and task not in ['questions-words.txt', 'questions-phrases.txt']:
            with open(f'../data/{args.dataset}/{task}', 'r') as f:
                stuff = f.read()
            categories[task[:-4]] = [l for l in stuff.split('\n')[1:] if l != '']
    
    # filter random empty strings
    categories = {k : [s for s in v if s != ''] for k, v in categories.items()}
    subfolders = ['concept', 'token', 'all', 'raw']

    if len(args.only_tasks) == 0:
        task_list = list(categories.keys())
    else:
        task_list = args.only_tasks

    # define constant prefixes for each of the tasks to disambiguate. 
    w_prefixes = {}
    if args.with_prefix:
        w_prefixes = {
            # prefixes for word2vec datasets 
            'capital-common-countries' : 'She travelled to ',
            'capital-world' : 'She travelled to ',
            'currency' : 'You will have to pay in ',
            'city-in-state' : 'She travelled to ',
            'family' : 'Did you talk to her ',
            'gram1-adjective-to-adverb' : 'Here is a random word in English: ',
            'gram2-opposite' : 'Here is a random word in English: ',
            'gram3-comparative' : 'Here is a random word in English: ',
            'gram4-superlative' : 'Here is a random word in English: ',
            'gram5-present-participle' : 'Here is a random word in English: ',
            'gram6-nationality-adjective' : 'Here is a random word in English: ', 
            'gram7-past-tense' : 'Here is a random word in English: ',
            'gram8-plural' : 'Here is a random word in English: ',
            'gram9-plural-verbs' : 'Here is a random word in English: ',
            
            # prefixes for fv datasets
            'antonym' : 'Here is a random word in English: ',
            'synonym' : 'Here is a random word in English: ',
            'present-past' : 'Here is a random word in English: ',
            'singular-plural' : 'Here is a random word in English: ',
            'word-length' : 'Here is a random word in English: ',
            'capitalize-first-letter' : 'Here is a random word/character: ',
            'capitalize-last-letter' : 'Here is a random word/character: ',
            'capitalize-second-letter' : 'Here is a random word/character: ',
            'lowercase-first-letter' : 'Here is a random word/character: ',
            'lowercase-last-letter' : 'Here is a random word/character: ',
            'next-capital-letter' : 'Here is a random word/character: ',
            'next-item' : 'Here is a random word/character: ',
            'prev-item' : 'Here is a random word/character: ',
            'capitalize' : 'Here is a random word in English: ',
            'country-capital' : 'She travelled to ',
            'country-currency' : 'You will have to pay in ',
            'english-french' : 'Voici un mot aléatoire en français: ',
            'english-german' : 'Hier ist ein beliebiges Wort im Deutschen: ',
            'english-spanish' : 'Aquí hay una palabra arbitraria en español: ',
            'landmark-country' : 'On vacation, we went to ',
            'national-parks' : 'On vacation, we went to ',
            'park-country' : 'On vacation, we went to ',
            'person-instrument' : 'I am a big fan of ',
            'person-occupation' : 'I am a big fan of ',
            'person-sport' : 'I am a big fan of ',
            'product-company' : 'I am a big fan of ', # eh 
            'sentiment' : "Here's my take on this film: ", 
        }
    else:
        w_prefixes = {task : '' for task in task_list}
    
    model = LanguageModel(args.model, device_map='cuda')

    # read in optimal layers for each task 
    optimal_layers = get_optimal_layers(task_list, dataset=args.dataset, with_prefix=args.with_prefix)

    for task in task_list:
        run_rank_scan(
            categories[task], task, model, optimal_layers[task][1], dataset=args.dataset, 
            concept_k=args.concept_k, token_k=args.token_k, w_prefix=w_prefixes[task]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', type=str)
    parser.add_argument('--dataset', default='word2vec', type=str)  # fvs
    parser.add_argument('--concept_k', default=80, type=int)
    parser.add_argument('--token_k', default=80, type=int)
    parser.add_argument('--only_tasks', nargs='+', default=[])
    parser.add_argument('--with_prefix', action='store_true')
    parser.set_defaults(with_prefix=False)
    args = parser.parse_args()
    main(args)