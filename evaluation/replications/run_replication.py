"""
Replication script for: Vector Arithmetic in Concept and Token Subspaces
This script runs the key experiments from the paper and compares with cached results.
"""

import os
os.chdir('/net/scratch2/smallyan/arithmetic_eval')

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from nnsight import LanguageModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Constants
MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
K = 80  # Number of top heads

# Head orderings to compare
HEAD_ORDERINGS = ['concept', 'token', 'all', 'raw']

# Key tasks to evaluate
KEY_TASKS = [
    'capital-common-countries',  # Semantic - concept should win
    'family',                    # Semantic - concept should win
    'gram5-present-participle',  # Grammatical - token should win
    'gram7-past-tense'           # Grammatical - token should win
]

# Layers to evaluate
KEY_LAYERS = [16, 20]

# Prefixes for tasks
TASK_PREFIXES = {
    'capital-common-countries': 'She travelled to ',
    'capital-world': 'She travelled to ',
    'currency': 'You will have to pay in ',
    'city-in-state': 'She travelled to ',
    'family': 'Did you talk to her ',
    'gram1-adjective-to-adverb': 'Here is a random word in English: ',
    'gram2-opposite': 'Here is a random word in English: ',
    'gram3-comparative': 'Here is a random word in English: ',
    'gram4-superlative': 'Here is a random word in English: ',
    'gram5-present-participle': 'Here is a random word in English: ',
    'gram6-nationality-adjective': 'Here is a random word in English: ',
    'gram7-past-tense': 'Here is a random word in English: ',
    'gram8-plural': 'Here is a random word in English: ',
    'gram9-plural-verbs': 'Here is a random word in English: ',
}

COLORS = {
    'all': 'green',
    'concept': 'indianred',
    'token': 'cornflowerblue',
    'raw': 'tab:orange'
}


def compute_logit_lens(vector, model):
    """Apply logit lens to a vector."""
    with torch.no_grad():
        normalized = model.model.norm(vector.cuda())
        logits = model.lm_head(normalized)
        probs = logits.softmax(dim=-1).detach().cpu()
    return probs


def build_ov_lens(model, head_ordering='concept', k=80, rank=4096):
    """Build the OV lens matrix by summing OV matrices from selected heads."""
    if head_ordering == 'raw':
        return None

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    model_name = model.config._name_or_path.split('/')[-1]

    if head_ordering == 'all':
        heads_to_sum = [(layer, head) for layer in range(num_layers)
                        for head in range(num_heads)]
    else:
        scores_path = f'cache/causal_scores/{model_name}/{head_ordering}_copying_len30_n1024.json'
        with open(scores_path, 'r') as f:
            head_scores = json.load(f)

        sorted_heads = sorted(head_scores, key=lambda x: x['score'], reverse=True)
        heads_to_sum = [(h['layer'], h['head_idx']) for h in sorted_heads[:k]]

    with torch.no_grad():
        ov_sum = torch.zeros((hidden_size, hidden_size), device='cuda')

        for layer, head in heads_to_sum:
            V = model.model.layers[layer].self_attn.v_proj.weight[
                head * head_dim : (head + 1) * head_dim
            ]
            O = model.model.layers[layer].self_attn.o_proj.weight[
                :, head * head_dim : (head + 1) * head_dim
            ]
            ov_sum += O @ V

        if rank < hidden_size:
            U, S, Vh = torch.linalg.svd(ov_sum)
            ov_sum = (U[:, :rank] * S[:rank]) @ Vh[:rank]

    return ov_sum


def extract_word_representation(word, model, layer, ov_lens=None, prefix=''):
    """Extract word representation at a given layer."""
    full_text = prefix + word.strip()

    with torch.no_grad():
        with model.trace(full_text):
            hidden_state = model.model.layers[layer].output[0].squeeze()[-1].detach().save()

    if ov_lens is None:
        return hidden_state
    else:
        return ov_lens @ hidden_state


def compute_word_embeddings(words, model, layer, ov_lens, prefix=''):
    """Compute embeddings for a set of words."""
    embeddings = {}
    for word in words:
        embeddings[word] = extract_word_representation(
            word, model, layer, ov_lens, prefix
        )
    return embeddings


def evaluate_parallelogram(a, b, a_prime, b_prime, embeddings, model):
    """Evaluate parallelogram arithmetic."""
    vec_a = embeddings[a]
    vec_b = embeddings[b]
    vec_a_prime = embeddings[a_prime]
    vec_b_prime = embeddings[b_prime]

    analogy_vec = vec_a - vec_b + vec_b_prime

    probs = compute_logit_lens(analogy_vec, model)
    pred_token = probs.argmax(dim=-1).item()
    pred_str = model.tokenizer.decode(pred_token)

    answer_token = model.tokenizer(a_prime)['input_ids'][1]
    answer_str = model.tokenizer.decode(answer_token)

    ll_correct = pred_str.strip().lower() == answer_str.strip().lower()
    ll_prob = probs[answer_token].item()

    ad_mean = (vec_a + vec_b_prime) / 2
    bc_mean = (vec_b + vec_a_prime) / 2
    score = torch.norm(ad_mean - bc_mean) / (torch.norm(vec_a - vec_b_prime) + torch.norm(vec_b - vec_a_prime))

    similarities = {}
    for word, vec in embeddings.items():
        similarities[word] = torch.cosine_similarity(analogy_vec, vec, dim=0).item()

    nearest = max(similarities, key=similarities.get)
    nn_correct = (nearest == a_prime)

    return ll_correct, ll_prob, score.item(), nn_correct


def load_task_data(task_name, dataset='word2vec'):
    """Load analogy task data."""
    file_path = f'data/{dataset}/{task_name}.txt'
    with open(file_path, 'r') as f:
        lines = f.read().split('\n')

    separator = '\t' if dataset == 'fvs' else ' '
    analogies = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split(separator)
            if len(parts) == 4:
                analogies.append(tuple(parts))

    return analogies


def get_unique_words(analogies):
    """Get all unique words from analogies."""
    words = set()
    for a, b, a_prime, b_prime in analogies:
        words.update([a, b, a_prime, b_prime])
    return words


def evaluate_task(task_name, model, layer, head_ordering, k=80, dataset='word2vec', prefix=''):
    """Evaluate a single task."""
    analogies = load_task_data(task_name, dataset)
    words = get_unique_words(analogies)

    ov_lens = build_ov_lens(model, head_ordering, k)
    embeddings = compute_word_embeddings(words, model, layer, ov_lens, prefix)

    ll_correct_count = 0
    nn_correct_count = 0
    ll_probs = []
    para_scores = []

    for a, b, a_prime, b_prime in analogies:
        ll_correct, ll_prob, para_score, nn_correct = evaluate_parallelogram(
            a, b, a_prime, b_prime, embeddings, model
        )

        ll_correct_count += ll_correct
        nn_correct_count += nn_correct
        ll_probs.append(ll_prob)
        para_scores.append(para_score)

    n = len(analogies)
    results = {
        'll_acc': ll_correct_count / n,
        'nn_acc': nn_correct_count / n,
        'n': n,
        'll_panswers': ll_probs,
        'parallelogram_scores': para_scores
    }

    return results


def load_cached_results(task, layer, ordering, dataset='word2vec', prefix_setting='with_prefix'):
    """Load cached results."""
    cache_path = f'cache/parallelograms/{dataset}/{prefix_setting}/{ordering}/{task}/layer{layer}_results.json'
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map='cuda', dispatch=True)
    print(f"Model loaded successfully")

    # Run evaluation
    results_with_prefix = defaultdict(lambda: defaultdict(dict))
    comparison_results = []

    for task in KEY_TASKS:
        print(f"\n=== Evaluating: {task} ===")
        prefix = TASK_PREFIXES.get(task, '')

        for layer in KEY_LAYERS:
            print(f"  Layer {layer}:")

            for ordering in HEAD_ORDERINGS:
                result = evaluate_task(
                    task, model, layer, ordering, k=K,
                    dataset='word2vec', prefix=prefix
                )
                results_with_prefix[task][layer][ordering] = result
                print(f"    {ordering}: NN acc = {result['nn_acc']:.3f}, LL acc = {result['ll_acc']:.3f}")

                # Compare with cached results
                cached = load_cached_results(task, layer, ordering)
                if cached:
                    diff = abs(result['nn_acc'] - cached['nn_acc'])
                    match = "MATCH" if diff < 0.01 else "DIFF"
                    comparison_results.append({
                        'task': task,
                        'layer': layer,
                        'ordering': ordering,
                        'replicated_nn_acc': result['nn_acc'],
                        'cached_nn_acc': cached['nn_acc'],
                        'difference': diff,
                        'match': match
                    })

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for cr in comparison_results:
        status = "✓" if cr['match'] == 'MATCH' else "✗"
        print(f"{status} {cr['task']} L{cr['layer']} {cr['ordering']}: "
              f"Replicated={cr['replicated_nn_acc']:.3f}, Cached={cr['cached_nn_acc']:.3f}, "
              f"Diff={cr['difference']:.4f}")

    # Calculate overall metrics
    matches = sum(1 for r in comparison_results if r['match'] == 'MATCH')
    total = len(comparison_results)
    avg_diff = np.mean([r['difference'] for r in comparison_results])

    print(f"\nOverall Match Rate: {matches}/{total} ({matches/total:.1%})")
    print(f"Average Accuracy Difference: {avg_diff:.4f}")

    # Print expected vs replicated
    print("\n" + "=" * 80)
    print("EXPECTED VS REPLICATED RESULTS")
    print("=" * 80)

    print("\nExpected Results (from Plan):")
    print("  Capital Cities (layer 20): Concept ~80%, Raw ~47%, Token ~20%")
    print("  Family (layer 20): Concept ~60%, Raw ~25%, Token ~10%")
    print("  Present Participle (layer 16): Token ~60%, Concept ~40%, Raw ~30%")
    print("  Past Tense (layer 16): Token ~65%, Concept ~45%, Raw ~35%")

    print("\nReplicated Results:")
    for task in KEY_TASKS:
        best_layer = 20 if task in ['capital-common-countries', 'family'] else 16
        if best_layer in results_with_prefix[task]:
            r = results_with_prefix[task][best_layer]
            print(f"  {task} (layer {best_layer}):")
            print(f"    Concept: {r['concept']['nn_acc']:.1%}")
            print(f"    Token:   {r['token']['nn_acc']:.1%}")
            print(f"    All:     {r['all']['nn_acc']:.1%}")
            print(f"    Raw:     {r['raw']['nn_acc']:.1%}")

    # Save results
    summary = {
        'tasks_evaluated': KEY_TASKS,
        'layers_evaluated': KEY_LAYERS,
        'head_orderings': HEAD_ORDERINGS,
        'k': K,
        'model': MODEL_NAME,
        'results': {task: {str(layer): {ordering: results_with_prefix[task][layer][ordering]
                                        for ordering in HEAD_ORDERINGS}
                           for layer in KEY_LAYERS}
                    for task in KEY_TASKS},
        'comparison': comparison_results,
        'overall_match_rate': matches / total,
        'average_difference': avg_diff
    }

    os.makedirs('evaluation/replications', exist_ok=True)
    with open('evaluation/replications/replication_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nResults saved to evaluation/replications/replication_summary.json")

    # Create visualization
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, task in enumerate(KEY_TASKS):
        ax = axes[idx // 2, idx % 2]

        best_layer = 20 if task in ['capital-common-countries', 'family'] else 16

        # Bar chart comparing orderings
        orderings = HEAD_ORDERINGS
        replicated_accs = [results_with_prefix[task][best_layer][o]['nn_acc'] for o in orderings]
        cached_accs = []
        for o in orderings:
            cached = load_cached_results(task, best_layer, o)
            cached_accs.append(cached['nn_acc'] if cached else 0)

        x = np.arange(len(orderings))
        width = 0.35

        bars1 = ax.bar(x - width/2, replicated_accs, width, label='Replicated',
                       color=[COLORS[o] for o in orderings], alpha=0.7)
        bars2 = ax.bar(x + width/2, cached_accs, width, label='Cached',
                       color=[COLORS[o] for o in orderings], alpha=0.4, hatch='//')

        ax.set_ylabel('NN Accuracy')
        ax.set_title(f'{task} (Layer {best_layer})')
        ax.set_xticks(x)
        ax.set_xticklabels(orderings)
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    plt.savefig('evaluation/replications/comparison_plot.png', dpi=300)
    print("Plot saved to evaluation/replications/comparison_plot.png")


if __name__ == '__main__':
    main()
