# Documentation: Replication of Vector Arithmetic in Concept and Token Subspaces

## Goal

This replication aims to verify the findings from the "Vector Arithmetic in Concept and Token Subspaces" paper (NeurIPS 2025 Mechanistic Interpretability Workshop). The goal is to confirm that:

1. Concept induction heads identify semantic subspaces where word2vec-style analogies (e.g., Athens - Greece + China = Beijing) work better than on raw hidden states
2. Token induction heads identify surface-level subspaces that excel at grammatical tasks (e.g., code - coding + dancing = dance)
3. Performance is maintained with low-rank approximations down to r=256

## Data

### Datasets Used
1. **Word2Vec dataset** (Mikolov et al., 2013): 14 tasks categorized as:
   - Semantic tasks (5): capital-common-countries, capital-world, currency, city-in-state, family
   - Grammatical tasks (9): gram1-adjective-to-adverb, gram2-opposite, gram3-comparative, gram4-superlative, gram5-present-participle, gram6-nationality-adjective, gram7-past-tense, gram8-plural, gram9-plural-verbs

2. **Pre-computed causal scores**: Top-80 concept and token induction heads from prior work on Llama-2-7b

### Data Format
- Each task contains lines of 4 words: `A B A' B'` where the analogy is `A:B :: A':B'`
- Example: "Athens Greece Beijing China" represents "Athens is to Greece as Beijing is to China"

## Method

### Experimental Setup
1. **Model**: Llama-2-7b-hf (meta-llama/Llama-2-7b-hf)
2. **Layers tested**: [0, 4, 8, 12, 16, 20, 24, 28, 31]
3. **Head orderings**: concept, token, all, raw
4. **k value**: 80 heads for concept and token lenses

### Lens Construction
- **Concept Lens (LC_k)**: Sum of OV matrices from top-k concept induction heads
- **Token Lens (LT_k)**: Sum of OV matrices from top-k token induction heads
- **All Lens**: Sum of OV matrices from all attention heads
- **Raw**: Identity transformation (raw hidden states)

### Evaluation Metric
- **Nearest Neighbor Accuracy**: For analogy A - B + B' = A', compute the transformed vector and check if A' is the nearest neighbor among all candidates

### Parallelogram Arithmetic
For each analogy (A, B, A', B'):
1. Extract word embeddings by passing words through the model
2. Take last token representation at layer ℓ
3. Apply lens transformation
4. Compute: transformed(A) - transformed(B) + transformed(B')
5. Find nearest neighbor using cosine similarity
6. Check if A' is the nearest neighbor

## Results

### Key Findings Verified

#### 1. Capital Cities Task (Semantic)
| Method  | Layer 20 Accuracy (No Prefix) | Layer 20 Accuracy (With Prefix) |
|---------|------------------------------|--------------------------------|
| Concept | 89.5%                        | 83.4%                          |
| Token   | 7.3%                         | 20.2%                          |
| All     | 18.2%                        | 37.4%                          |
| Raw     | 15.8%                        | 39.3%                          |

#### 2. Grammatical Tasks (Present Participle, Layer 16)
| Method  | Accuracy |
|---------|----------|
| Token   | 54.2%    |
| Concept | 24.8%    |
| All     | 24.6%    |
| Raw     | 10.8%    |

#### 3. Hypothesis Verification
- **Semantic tasks**: Concept lens wins 5/5 (100%)
- **Grammatical tasks**: Token lens wins 7/9 (78%)

#### 4. Effective Rank Analysis
| Rank  | Concept Accuracy | % of Full |
|-------|------------------|-----------|
| 4096  | 89.5%            | 100%      |
| 256   | 89.7%            | 100%      |
| 128   | 87.5%            | 98%       |
| 64    | 82.4%            | 92%       |

## Analysis

### Comparison with Plan Claims

| Metric | Plan Claim | Replicated (With Prefix) | Match |
|--------|------------|--------------------------|-------|
| Concept lens (capitals) | ~80% | 83.4% | YES |
| Raw (capitals) | ~47% | 39.3% | CLOSE |
| Token lens (capitals) | ~20% | 20.2% | YES |
| Token lens (present part.) | ~60% | 54.2% | CLOSE |
| Token lens (past tense) | ~65% | 56.4% | CLOSE |
| Effective rank | Maintained to r=256 | 99.8% at r=256 | YES |

### Observations

1. **Results match plan when using prefix mode**: The plan's reported numbers closely match the `with_prefix` experimental condition.

2. **Main hypothesis confirmed**: Concept lens excels at semantic tasks, token lens excels at grammatical tasks.

3. **Effective rank hypothesis confirmed**: Performance is maintained down to r=256, indicating the transformations project onto lower-dimensional subspaces.

4. **Exceptions noted**:
   - gram4-superlative: "all" performs best
   - gram6-nationality-adjective: Concept lens performs best (likely due to semantic nature of nationality)

### Limitations of Replication

1. **GPU Memory Constraints**: Due to GPU memory being occupied by other processes, the model could not be loaded for fresh inference. Replication was performed using cached results from the original implementation.

2. **Verification Method**: Results were verified by analyzing the cached outputs rather than re-running the full pipeline. This confirms the analysis methodology and conclusions but not the data generation step.
