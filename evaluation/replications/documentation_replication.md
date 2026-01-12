# Documentation: Replication of "Vector Arithmetic in Concept and Token Subspaces"

## Goal

This replication aims to verify the experimental results from the paper "Vector Arithmetic in Concept and Token Subspaces" (NeurIPS 2025 Mechanistic Interpretability Workshop) by Sheridan Feucht, Byron Wallace, and David Bau.

The core hypothesis is that concept and token induction heads can identify subspaces of Llama-2-7b activations with coherent semantic and surface-level structure, enabling more accurate parallelogram (word2vec-style) arithmetic than using raw hidden states.

## Data

### Datasets Used

1. **Word2Vec Dataset** (`data/word2vec/`)
   - Contains 14 analogy tasks from Mikolov et al. (2013)
   - Semantic tasks: capital-common-countries, capital-world, currency, city-in-state, family
   - Grammatical tasks: gram1-adjective-to-adverb through gram9-plural-verbs
   - Format: Each line contains 4 words forming an analogy (a:b :: a':b')

2. **Function Vector Tasks Dataset** (`data/fvs/`)
   - Contains 23 semantic and grammatical transformation tasks
   - Includes translations, antonyms/synonyms, and surface transformations

### Pre-computed Resources

1. **Head Importance Scores** (`cache/causal_scores/Llama-2-7b-hf/`)
   - `concept_copying_len30_n1024.json`: Concept head copying scores
   - `token_copying_len30_n1024.json`: Token head copying scores
   - Used to identify top-k heads for concept/token lenses

## Method

### Methodology Overview

1. **Build Concept and Token Lenses**
   - Sum OV matrices (O_l,h × V_l,h) from top-k concept/token induction heads
   - Creates transformation matrices L_C^k and L_T^k
   - k=80 heads used as default

2. **Extract Word Embeddings**
   - Pass single words through Llama-2-7b
   - Extract last token representation at layer ℓ
   - Optionally apply task-specific prefixes (e.g., "She travelled to " for capital cities)
   - Transform using lens matrices: L × hidden_state

3. **Test Parallelogram Arithmetic**
   - For word tuples (a,b) and (a',b'), compute: L(a) - L(b) + L(b')
   - Measure if L(a') is the nearest neighbor among all candidate words
   - Primary metric: Nearest-neighbor accuracy

4. **Compare Four Settings**
   - **raw**: Use raw hidden states (L = Identity)
   - **concept**: Use concept lens (L = L_C^k)
   - **token**: Use token lens (L = L_T^k)
   - **all**: Use all attention heads (L = L_all)

### Key Functions Implemented

1. `build_ov_lens()`: Constructs the OV lens matrix by:
   - Loading head scores from pre-computed JSON files
   - Sorting heads by score and selecting top-k
   - Summing O @ V matrices for selected heads
   - Optionally applying low-rank approximation via SVD

2. `extract_word_representation()`: Gets word vectors by:
   - Passing text (with optional prefix) through the model
   - Extracting hidden state at specified layer
   - Applying OV lens transformation

3. `evaluate_parallelogram()`: Evaluates analogy by:
   - Computing analogy vector: vec(a) - vec(b) + vec(b')
   - Finding nearest neighbor via cosine similarity
   - Computing logit lens accuracy (secondary metric)

## Results

### Key Tasks Evaluated

Four representative tasks were evaluated across layers 16 and 20:

| Task | Best Layer | Concept | Token | All | Raw |
|------|-----------|---------|-------|-----|-----|
| capital-common-countries | 20 | **83.4%** | 20.2% | 37.4% | 39.3% |
| family | 20 | **51.6%** | 10.7% | 34.6% | 19.2% |
| gram5-present-participle | 16 | 48.3% | **68.3%** | 49.1% | 30.1% |
| gram7-past-tense | 16 | 52.9% | **85.4%** | 53.1% | 31.9% |

### Comparison with Expected Results (from Plan)

| Task | Expected | Replicated | Match |
|------|----------|------------|-------|
| Capital Cities - Concept | ~80% | 83.4% | Yes |
| Capital Cities - Raw | ~47% | 39.3% | Close |
| Capital Cities - Token | ~20% | 20.2% | Yes |
| Family - Concept | ~60% | 51.6% | Close |
| Family - Raw | ~25% | 19.2% | Close |
| Family - Token | ~10% | 10.7% | Yes |
| Present Participle - Token | ~60% | 68.3% | Yes |
| Present Participle - Concept | ~40% | 48.3% | Close |
| Past Tense - Token | ~65% | 85.4% | Better |
| Past Tense - Concept | ~45% | 52.9% | Close |

### Replication Accuracy

- **100% match** with cached results (32/32 comparisons)
- Average accuracy difference: 0.0000
- The replication exactly reproduces the original implementation

## Analysis

### Key Findings Confirmed

1. **Concept lens excels at semantic tasks**:
   - Capital cities: 83.4% (vs 39.3% raw) - 44.1 percentage point improvement
   - Family: 51.6% (vs 19.2% raw) - 32.4 percentage point improvement

2. **Token lens excels at grammatical tasks**:
   - Present participle: 68.3% (vs 30.1% raw) - 38.2 percentage point improvement
   - Past tense: 85.4% (vs 31.9% raw) - 53.5 percentage point improvement

3. **Raw hidden states consistently underperform**:
   - Supports hypothesis that interference from irrelevant information degrades parallelogram arithmetic

4. **Layer-dependent performance**:
   - Semantic tasks peak around layer 20
   - Grammatical tasks peak around layer 16

### Discrepancies with Plan

The plan mentioned specific accuracy values that differ slightly from both replicated and cached results:
- Capital Cities: Plan said "~80% at layer 20" for concept lens, actual is 83.4%
- Past Tense: Plan said "~65% at layer 16" for token lens, actual is 85.4%

These discrepancies may be due to:
- Rounded figures in the plan
- Different experimental settings (e.g., with/without prefix)
- The plan may have described preliminary results

### Special Cases and Notes

- No external API keys were required
- Model loaded successfully from shared cache
- All pre-computed head scores were available
- Environment was fully reproducible with standard packages

## Artifacts Generated

1. `replication.ipynb` - Full Jupyter notebook with reimplemented code
2. `run_replication.py` - Standalone Python script for replication
3. `replication_summary.json` - Detailed results in JSON format
4. `comparison_plot.png` - Visualization of replicated vs cached results
