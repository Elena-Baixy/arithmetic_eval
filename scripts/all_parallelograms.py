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
import argparse
from nnsight import LanguageModel
from parallelograms import get_neighbors, calculate_save_scores

def loop_for_task(this_task, task_name, model, subfolders, layers, concept_k, token_k, w_prefix, dataset):
    sep = ' ' if dataset == 'word2vec' else '\t'
    print(task_name, w_prefix, this_task[0].split(sep)[0])
    w_prefixes = (w_prefix, w_prefix) # just always keep left prefix = right prefix. 
    for head_ordering in subfolders:
        for layer in layers:
            k = token_k if head_ordering == 'token' else concept_k

            # for this head_ordering+layer, get representations for all the neighbors and calculate scores. 
            neighbors = get_neighbors(
                this_task, model, layer, head_ordering, k, w_prefixes, dataset, rank=model.config.hidden_size
            )
            calculate_save_scores(
                this_task, neighbors, model, k, head_ordering, dataset, task_name, layer, w_prefixes, rank=4096
            )            
            del neighbors

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
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]

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

    model = LanguageModel(args.model, device_map='cuda', dispatch=True)

    for task in task_list:
        loop_for_task(
            categories[task], task, model, subfolders, layers, dataset=args.dataset, 
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