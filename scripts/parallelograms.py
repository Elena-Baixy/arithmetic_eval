''' 
At a particular layer, take word2vec style parallelograms for:
    - sum of concept head OVs lm_head(OV(Tokyo_ell) - OV(Japan_ell) + OV(Germany_ell)) = Berlin?
    - sum of token head OVs (same thing)
    - sum of all the head OVs (same thing)
    - raw hidden states at last token position 

Summing all the head OVs is a nice comparison bc you get that concept signal still but it's surely drowned out
by all the other things heads could be reading from that hidden state + contributing to resid. at a given time.
'''
import os 
import torch 
import json 
import argparse
import matplotlib.pyplot as plt 
from nnsight import LanguageModel

def logit_lens(concept_vec, model):
    with torch.no_grad():
        return model.lm_head(model.model.norm(concept_vec.cuda())).softmax(dim=-1).detach().cpu() # vocab_size 

def print_logit_lens(probs, tokenizer, label=''):
    topprobs, idxs = torch.topk(probs, k=10)
    print(f'{label} logit lens\t', [(tokenizer.decode(t), round(p.item(), 3)) for t, p in zip(idxs, topprobs)])

# take a word, pass it thru the network, and pass thru summed OV matrix for top-k concept heads.
def proj_onto_ov(w, ov_sum, model, layer_idx, head_ordering='concept', offset=-1, w_prefix=''):
    # add space in front of word to avoid weird tokenization
    # or other things if we want context for the word 
    w = w_prefix + w.strip()

    # just return raw hidden state if 'raw'
    if head_ordering == 'raw':
        with torch.no_grad(), model.trace(w):
            state = model.model.layers[layer_idx].output[0].squeeze()[offset].save()
        return state 

    # otherwise apply OV matrix to state
    with torch.no_grad():
        with model.trace(w):
            state = model.model.layers[layer_idx].output[0].squeeze()[offset].detach().save()
    return torch.matmul(ov_sum, state)

def get_ov_sum(model, head_ordering='concept', k=80, rank=4096):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    model_name = model.config._name_or_path.split('/')[-1]
    
    if head_ordering == 'raw':
        return None
    elif head_ordering == 'all':
        to_sum = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(model.config.num_attention_heads)]
    else: 
        with open(f'../cache/causal_scores/{model_name}/{head_ordering}_copying_len30_n1024.json', 'r') as f: 
            temp = json.load(f)
        tups = sorted([(d['layer'], d['head_idx'], d['score']) for d in temp], key=lambda t: t[2], reverse=True)
        to_sum = [(l, h) for l, h, _ in tups][:k]
    layerset = set([l for l, _ in to_sum])

    # get our actual OV matrix 
    with torch.no_grad():
        ov_sum = torch.zeros((4096, 4096), device='cuda')
        for layer in layerset:
            for l, h in to_sum:
                if l == layer:
                    # (out_features, in_features). 
                    V = model.model.layers[l].self_attn.v_proj.weight[h * head_dim : (h+1) * head_dim] # select rows so that (128, 4096) projects hidden state down. 
                    O = model.model.layers[l].self_attn.o_proj.weight[:, h * head_dim : (h+1) * head_dim] # select columns so that (4096, 128) converts value back up
                    ov_sum += torch.matmul(O, V) # 4096, 4096
        
        # reduce rank if desired
        if rank < model.config.hidden_size:
            U, S, Vh = torch.linalg.svd(ov_sum)
            ov_sum = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        return ov_sum 

# for this task, get representations for all the neighbors.
def get_neighbors(task_lines, model, layer, head_ordering, k, w_prefixes, dataset, rank):
    sep = ' ' if dataset == 'word2vec' else '\t'
    ov_sum = get_ov_sum(model, head_ordering, k, rank)

    # if these guys all take the same prefix (this is what we do in the paper)
    if w_prefixes[0] == w_prefixes[1]:
        neighbors = set([w for l in task_lines for w in l.split(sep)])
        neighbors = {
            w : proj_onto_ov(w, ov_sum, model, layer, head_ordering=head_ordering, w_prefix=w_prefixes[0])
            for w in neighbors  
        } # keep `offset` at -1 to get the last token representation of this word.

    # we might also want to give diff prefixes e.g. "She travelled to the country of {Japan/India/China}" vs. 
    # "She travelled to the city of {Calgary/Paris/Delhi}". we don't actually do this in the paper 
    else: 
        left_neighbors = set([l.split(sep)[0] for l in task_lines])
        right_neighbors = set([l.split(sep)[1] for l in task_lines])
        neighbors = {}
        for w in left_neighbors:
            neighbors[w] = proj_onto_ov(w, model, layer, head_ordering=head_ordering, k=k, w_prefix=w_prefixes[0])
        for w in right_neighbors:
            neighbors[w] = proj_onto_ov(w, model, layer, head_ordering=head_ordering, k=k, w_prefix=w_prefixes[1])

    return neighbors

# return: exact acc, P(answer), parallelogram score 
def get_parallelogram_scores(a, b, c, d, neighbors, model, verbose=False):
    aw, bw, cw, dw = a, b, c, d
    a = neighbors[aw] # retrieve pre-calculated vectors for each word 
    b = neighbors[bw]
    c = neighbors[cw]
    d = neighbors[dw]

    # answer token should be 'Hav' for 'Havana', lowered to eliminate caps. 
    ans_tok = model.tokenizer(cw)['input_ids'][1] # skip bos 
    ans_str = model.tokenizer.decode(ans_tok)

    # calculate logit lens match and P(answer)
    probs = logit_lens((a - b) + d, model)
    pred = model.tokenizer.decode(probs.argmax(dim=-1))

    ll_correct = pred.strip().lower() == ans_str.strip().lower()
    ll_pans = probs[ans_tok].item()

    # calculate parallelogram score (unused in paper)
    admean = (a + d) / 2
    bcmean = (b + c) / 2
    score = torch.norm(admean - bcmean) / (torch.norm(a - d) + torch.norm(b - c))
    
    # calculate nearest neighbor scores 
    similarities = {}
    for k in neighbors.keys():
        similarities[k] = torch.cosine_similarity((a - b) + d, neighbors[k], dim=0)
    nn_correct = max(similarities, key=similarities.get) == cw        
    if verbose:
        print(f'{aw} - {bw} + {dw} : {cw}?', pred, ll_correct, f'parallel_score={round(score.item(), 3)}') 
        print('neighbors:', sorted(similarities, key=similarities.get, reverse=True)[:5])

    return ll_correct, ll_pans, score.item(), nn_correct

def all_dot_products(task_lines, neighbors, model, k, head_ordering, dataset, task_name, layer, w_prefixes, rank):
    sep = ' ' if dataset == 'word2vec' else '\t'
    # for all lines [Moscow Russia Berlin Germany], calculate
    # the dot product and add to list. 
    dots = []
    cosines = []
    for line in task_lines:
        if len(line.split(sep)) == 4:
            a, b, aprime, bprime = line.split(sep)
            a = neighbors[a] # retrieve pre-calculated vectors for each word 
            b = neighbors[b]
            aprime = neighbors[aprime]
            bprime = neighbors[bprime]

            # (a - b) \cdot (a' - b')
            dots.append(
                torch.dot(a - b, aprime - bprime).item()
            )
            cosines.append(
                torch.cosine_similarity(a - b, aprime - bprime, dim=0).item()
            )

    if w_prefixes[0] == '' and w_prefixes[1] == '':
        superfolder = 'no_prefix'
    else:
        superfolder = 'with_prefix'
    os.makedirs(f'../cache/parallelograms/{dataset}/{superfolder}/{head_ordering}/{task_name}', exist_ok=True)
    os.makedirs(f'../figures/parallelograms/{dataset}/{superfolder}/{task_name}', exist_ok=True)

    fname = f'layer{layer}'
    fname += f'_rank{rank}' if rank < model.config.hidden_size else ''

    results = {
        'dots' : dots,
        'cosines' : cosines 
    }
    with open(f'../cache/parallelograms/{dataset}/{superfolder}/{head_ordering}/{task_name}/{fname}_dots.json', 'w') as f:
        json.dump(results, f)

    colors = {
        'all' : 'green',
        'concept' : 'indianred',
        'token' : 'cornflowerblue',
        'raw' : 'tab:orange'
    }
    plt.hist(dots, color=colors[head_ordering], edgecolor='black')
    plt.title(f'All Possible {task_name} Dot Products')
    plt.ylabel('Count')
    plt.xlabel('Dot Product of Diff. Pair (e.g. (man - woman) * (king - queen))')
    plt.savefig(f'../figures/parallelograms/{dataset}/{superfolder}/{task_name}/{head_ordering}_{fname}_dot_hist.png')
    plt.clf()

    plt.hist(cosines, color=colors[head_ordering], edgecolor='black')
    plt.title(f'All Possible {task_name} Cosine Similarities')
    plt.ylabel('Count')
    plt.xlabel('Cosine Sim. of Diff. Pair (e.g. (man - woman) * (king - queen))')
    plt.xlim(-1, 1)
    plt.savefig(f'../figures/parallelograms/{dataset}/{superfolder}/{task_name}/{head_ordering}_{fname}_cosine_hist.png')
    plt.clf()


def calculate_save_scores(task_lines, neighbors, model, k, head_ordering, dataset, task_name, layer, w_prefixes, rank):
    sep = ' ' if dataset == 'word2vec' else '\t'
    # for all [Moscow Russia Berlin Germany] examples, calculate: 
    # exact parallelogram accuracy, parallelogram score, top1_prob
    # for concept OV, token OV, all OV 
    ll_acc = 0; n = 0
    panswers = []
    parallelogram_scores = []
    nn_acc = 0 
    for line in task_lines:
        if len(line.split(sep)) == 4:
            a, b, aprime, bprime = line.split(sep)
            # print(a, 'is to', b, 'as', aprime, 'is to', bprime)

            # a - b + bprime = aprime 
            ll_corr, ll_pans, score, nn_corr = get_parallelogram_scores(
                a, b, aprime, bprime, neighbors, model, verbose=False
            ) 

            # save info 
            ll_acc += ll_corr 
            n += 1 
            panswers.append(ll_pans)
            parallelogram_scores.append(score)
            nn_acc += nn_corr

    # calculate accuracy and print
    ll_acc /= n
    nn_acc /= n 
    print(head_ordering, task_name, 'layer', layer)
    print('logit lens accuracy', ll_acc)
    print('nearest neighbor accuracy', nn_acc)
    print('average P(aprime)', sum(panswers) / len(panswers))
    print('average parallelogram score', sum(parallelogram_scores) / len(parallelogram_scores))

    # save overall json 
    results = {
        'll_acc' : ll_acc,
        'nn_acc' : nn_acc,
        'n' : n,
        'll_panswers' : panswers,
        'parallelogram_scores' : parallelogram_scores,
    }

    if w_prefixes[0] == '' and w_prefixes[1] == '':
        superfolder = 'no_prefix'
    else:
        superfolder = 'with_prefix'
    os.makedirs(f'../cache/parallelograms/{dataset}/{superfolder}/{head_ordering}/{task_name}', exist_ok=True)

    fname = f'layer{layer}'
    fname += f'_rank{rank}' if rank < model.config.hidden_size else ''
    fname += '_results.json'

    with open(f'../cache/parallelograms/{dataset}/{superfolder}/{head_ordering}/{task_name}/{fname}', 'w') as f:
        json.dump(results, f)

def main(args):
    # load in the relevant task 
    with open(f'../data/{args.dataset}/{args.task}.txt', 'r') as f:
        stuff = f.read()
    this_task = [l for l in stuff.split('\n')[1:] if l != '']

    print(args.w_prefix)
    if type(args.w_prefix) == str:
        w_prefixl, w_prefixr = args.w_prefix, args.w_prefix
    elif type(args.w_prefix) == list:
        if len(args.w_prefix) == 1:
            w_prefixl, w_prefixr = args.w_prefix[0], args.w_prefix[0]
        else:
            assert len(args.w_prefix) == 2
            w_prefixl, w_prefixr = args.w_prefix[0], args.w_prefix[1]
    sep = '\t' if args.dataset == 'fvs' else ' '
    example_l, example_r, _, _ = this_task[0].split(sep)
    print(w_prefixl + example_l, w_prefixr + example_r)

    # load model 
    model = LanguageModel(args.model, device_map='cuda', dispatch=True)

    # get neighbor representations and calculate scores  
    neighbors = get_neighbors(this_task, model, args.layer, args.head_ordering, args.k, (w_prefixl, w_prefixr), args.dataset, args.rank)
    calculate_save_scores(
        this_task, neighbors, model, args.k, args.head_ordering, args.dataset, args.task, args.layer, (w_prefixl, w_prefixr), args.rank
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', type=str)
    parser.add_argument('--head_ordering', default='concept', choices=['concept', 'token', 'all', 'raw'], type=str)
    parser.add_argument('--dataset', default='word2vec', type=str) # fvs
    parser.add_argument('--task', default='capital-common-countries', type=str) 
    parser.add_argument('--w_prefix', default='', type=str, nargs='+') 
    parser.add_argument('--layer', default=20, type=int)
    parser.add_argument('--k', default=80, type=int)
    parser.add_argument('--rank', default=4096, type=int)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    main(args)

