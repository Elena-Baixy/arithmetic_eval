# Replication Documentation: Vector Arithmetic in Concept and Token Subspaces

## Goal

Replicate the experiments from "Vector Arithmetic in Concept and Token Subspaces" demonstrating that concept and token induction heads can identify subspaces of Llama-2-7b activations with coherent semantic and surface-level structure, enabling more accurate parallelogram arithmetic (e.g., Athens - Greece + China = Beijing) than using raw hidden states.

## Data

### Datasets Used
1. **word2vec** - Original word analogy data from Mikolov et al. (2013)
   - `capital-common-countries.txt`: 506 examples of capital-country pairs
   - `family.txt`: 506 examples of family relations
   - `gram5-present-participle.txt`: 1056 examples of present participle transformations
   - `gram7-past-tense.txt`: 1560 examples of past tense transformations

### Pre-computed Resources
- **Causal scores**: Top-k concept and token induction heads from `cache/causal_scores/Llama-2-7b-hf/`
  - `concept_copying_len30_n1024.json`: Scores for concept heads
  - `token_copying_len30_n1024.json`: Scores for token heads

## Method

### 1. Build OV Lenses
- Load causal scores for concept/token induction heads
- Select top-k heads (k=80 by default) based on copying scores
- Sum OV matrices (O @ V) across selected heads to create transformation lens

### 2. Extract Word Representations
- Pass words through Llama-2-7b-hf
- Extract hidden state at specified layer (last token position)
- Optionally transform through OV lens matrix

### 3. Parallelogram Arithmetic
- For each tuple (a, b, a', b'), compute: a - b + b'
- Measure nearest neighbor accuracy: Is a' the nearest neighbor?
- Compare across methods: raw, concept lens, token lens, all heads

### 4. Layer Sweep
- Test layers [0, 4, 8, 12, 16, 20, 24, 28, 31]
- Identify optimal layers for each task type

## Results

### Capital Cities (Layer 20)
| Method | NN Accuracy |
|--------|-------------|
| Raw | 15.81% |
| Concept | **89.53%** |
| Token | 7.31% |
| All | 18.18% |

### Family Relations (Layer 20)
| Method | NN Accuracy |
|--------|-------------|
| Raw | 0.40% |
| Concept | **6.92%** |
| Token | 2.37% |
| All | 3.36% |

### Present Participle (Layer 16)
| Method | NN Accuracy |
|--------|-------------|
| Raw | 10.80% |
| Concept | 24.72% |
| Token | **54.17%** |
| All | 24.72% |

### Past Tense (Layer 16)
| Method | NN Accuracy |
|--------|-------------|
| Raw | 9.49% |
| Concept | 25.58% |
| Token | **56.41%** |
| All | 30.00% |

## Analysis

### Key Findings

1. **Concept Lens Excellence on Semantic Tasks**: The concept lens dramatically outperforms raw hidden states on semantic analogy tasks like capital-country relations (89.5% vs 15.8%). This supports the hypothesis that concept induction heads operate in a semantic subspace.

2. **Token Lens Excellence on Grammatical Tasks**: The token lens achieves best performance on morphological/grammatical tasks like present participle (54.2%) and past tense (56.4%), outperforming both raw states and concept lens. This confirms that token heads capture surface-level/wordform information.

3. **Layer Dependence**: Optimal performance varies by task:
   - Semantic tasks peak around layers 16-20
   - All methods show poor performance at very early (0) and very late (31) layers

4. **All-Heads Baseline**: Simply using all attention heads does not recover the specialized benefits of concept/token lenses, suggesting the top-k selection is crucial.

### Validation

All replicated results exactly match the original cached values from the repository, with zero discrepancy across all tasks, methods, and layers tested.

## Reproducibility Notes

- Model: `meta-llama/Llama-2-7b-hf` with float16 precision
- GPU: NVIDIA A40 (47.7 GB)
- Framework: nnsight for model tracing
- Random seeds: Not required (deterministic operations)
