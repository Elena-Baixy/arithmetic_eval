# Replication Documentation: Vector Arithmetic in Concept and Token Subspaces

## Goal

Replicate the core experiments from the paper "Vector Arithmetic in Concept and Token Subspaces" (NeurIPS 2025 Mechanistic Interpretability Workshop) to verify that:

1. Concept induction heads identify semantic subspaces enabling accurate word2vec-style parallelogram arithmetic
2. Token induction heads identify surface-level subspaces for grammatical transformations
3. Both lens types outperform raw hidden states for their respective task categories

## Data

### Model
- **Llama-2-7b-hf** (meta-llama/Llama-2-7b-hf)
- Hidden size: 4096
- Number of attention heads: 32
- Number of layers: 32

### Datasets
- **word2vec dataset**: Parallelogram arithmetic tasks from Mikolov et al. (2013)
  - `capital-common-countries`: Semantic task (e.g., Athens - Greece + China = Beijing)
  - `family`: Semantic task (e.g., son - daughter + mom = dad)
  - `gram5-present-participle`: Grammatical task (e.g., code - coding + dancing = dance)
  - `gram7-past-tense`: Grammatical task (e.g., coding - coded + danced = dancing)

### Pre-computed Resources
- Concept head rankings from causal scores (`concept_copying_len30_n1024.json`)
- Token head rankings from causal scores (`token_copying_len30_n1024.json`)

## Method

### 1. Build Concept and Token Lenses
Sum OV matrices (O_l,h * V_l,h) from top-k concept/token induction heads:
- **k = 80** heads used for both concept and token lenses
- OV matrices extracted from model attention layers
- Sum computed as: `ov_sum = sum(O @ V for (layer, head) in top_k_heads)`

### 2. Extract Word Embeddings
For each word in a parallelogram task:
1. Pass word through model
2. Extract hidden state at specified layer (last token position)
3. Transform using lens matrix: `transformed = ov_sum @ hidden_state`

### 3. Parallelogram Arithmetic Evaluation
For tuples (a, b, a', b') where a - b should equal a' - b':
1. Compute result vector: `result = transformed_a - transformed_b + transformed_b'`
2. **Nearest Neighbor Accuracy**: Check if a' is the nearest neighbor to result (cosine similarity)
3. **Logit Lens Accuracy**: Check if applying lm_head predicts the correct first token

### 4. Settings Compared
- **Concept lens**: Top 80 concept induction heads
- **Token lens**: Top 80 token induction heads
- **Raw**: No transformation (identity matrix)
- Layers evaluated: [8, 12, 16, 20, 24]

## Results

### Capital Cities Task (capital-common-countries)
| Layer | Concept NN Acc | Token NN Acc | Raw NN Acc |
|-------|----------------|--------------|------------|
| 8     | 34.4%          | 3.4%         | 5.1%       |
| 12    | 61.1%          | 5.3%         | 9.1%       |
| 16    | 87.9%          | 6.1%         | 17.4%      |
| 20    | **89.5%**      | 6.9%         | 16.0%      |
| 24    | 86.6%          | 3.4%         | 5.9%       |

### Family Relations Task (family)
| Layer | Concept NN Acc | Token NN Acc | Raw NN Acc |
|-------|----------------|--------------|------------|
| 8     | 2.2%           | 1.2%         | 0.4%       |
| 12    | 3.2%           | 0.8%         | 0.2%       |
| 16    | 2.4%           | 2.2%         | 0.6%       |
| 20    | **6.7%**       | 2.4%         | 0.4%       |
| 24    | 4.7%           | 1.2%         | 0.2%       |

### Present Participle Task (gram5-present-participle)
| Layer | Concept NN Acc | Token NN Acc | Raw NN Acc |
|-------|----------------|--------------|------------|
| 8     | 3.5%           | 11.5%        | 2.6%       |
| 12    | 15.5%          | 36.3%        | 5.7%       |
| 16    | 23.4%          | **54.8%**    | 11.0%      |
| 20    | 9.5%           | 40.4%        | 2.7%       |
| 24    | 2.7%           | 20.2%        | 0.9%       |

### Past Tense Task (gram7-past-tense)
| Layer | Concept NN Acc | Token NN Acc | Raw NN Acc |
|-------|----------------|--------------|------------|
| 8     | 2.7%           | 11.6%        | 1.6%       |
| 12    | 9.7%           | 25.3%        | 4.1%       |
| 16    | 25.5%          | **56.0%**    | 9.4%       |
| 20    | 13.5%          | 39.4%        | 3.8%       |
| 24    | 7.1%           | 19.7%        | 1.4%       |

### Comparison with Original Results
- **Total comparisons**: 60 (4 tasks x 3 orderings x 5 layers)
- **Mean absolute difference**: 0.24%
- **Maximum absolute difference**: 1.42%
- **All differences within 2%**: Yes

## Analysis

### Key Findings Validated

1. **Semantic Task Performance**: Concept lens dramatically outperforms raw hidden states for semantic tasks (capital cities: 89.5% vs 16.0%), confirming that concept induction heads identify semantically coherent subspaces.

2. **Grammatical Task Performance**: Token lens significantly outperforms concept lens for grammatical tasks (past tense: 56.0% vs 25.5%), validating that token induction heads capture surface-level word structure.

3. **Layer Dependency**: Optimal performance occurs at middle-to-late layers (16-20), consistent with the paper's findings about when semantic and surface information becomes most accessible.

4. **Replication Accuracy**: All results match original cached values within 2% difference, demonstrating high reproducibility.

### Limitations

1. **Family Task**: Low overall accuracy (~7% for concept lens) suggests this task is inherently more difficult, possibly due to higher ambiguity in family relationship representations.

2. **Subset Evaluation**: Only evaluated on 4 of 14 word2vec tasks and 5 of 9 layers due to time constraints.

### Conclusion

The replication successfully validates the paper's central claims about the complementary roles of concept and token induction heads in enabling word2vec-style vector arithmetic on language model activations.
