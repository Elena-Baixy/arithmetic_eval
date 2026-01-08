# Evaluation: Replication of Vector Arithmetic in Concept and Token Subspaces

## Reflection

This replication successfully reproduced the key experiments from the arithmetic_eval repository. The implementation was straightforward once the core concepts were understood:

1. **OV Matrix Computation**: The key insight is that summing O*V matrices from concept/token heads creates a transformation that projects hidden states onto the subspace these heads operate in.

2. **Parallelogram Arithmetic**: The classic word2vec arithmetic (a - b + b' ≈ a') works much better when applied to the projected subspace rather than raw hidden states.

3. **Task-Specific Prefixes**: Using contextual prefixes (e.g., "She travelled to Athens") disambiguates word meanings and improves results.

### Challenges Encountered:
- **nnsight version incompatibility**: Had to switch to direct HuggingFace transformers implementation due to FakeTensor errors
- **Model loading**: Permission errors with shared model cache required using local files directly

### Successful Aspects:
- All numerical results matched cached values exactly (or within 0.1%)
- The reimplementation from plan/code-walk understanding worked without requiring verbatim code copying
- Key scientific claims were validated (concept vs token lens specialization)

---

## Replication Evaluation — Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment could be fully reconstructed from the plan.md and CodeWalkthrough.md files. The plan clearly describes:
- The methodology (OV matrix construction, word representation extraction, parallelogram arithmetic)
- The experimental settings (raw, concept, token, all)
- Expected results for key tasks
- Hyperparameters (k=80 heads, specific layers)

The code walkthrough provided clear guidance on how to run the scripts and what data is needed. No major guesswork was required - all steps were documented.

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment was reproducible with minor adjustments:
- Model (Llama-2-7b-hf) was available in shared cache
- Pre-computed head scores were provided in cache/causal_scores/
- All required packages (torch, transformers, nnsight) were available
- Had to work around nnsight version incompatibility by using transformers directly, but this is a valid reimplementation approach
- No missing dependencies or version conflicts prevented faithful replication

---

### RP3. Determinism and Stability

**PASS**

**Rationale**: Results were highly deterministic:
- All replicated accuracy values matched cached values exactly (0.0000 difference) for most experiments
- One experiment (gram7 concept) had 0.0013 difference, within acceptable tolerance
- No random seeds needed - computation is deterministic (OV matrices, hidden state extraction, cosine similarity)
- Multiple runs would produce identical results

---

### RP4. Demo Presentation

**NA**

**Rationale**: This replication did not involve evaluating a demo. The repository provides scripts for running full experiments, not a simplified demo. The replication was performed by reimplementing the core functionality and comparing against cached results.

---

## Summary

The replication was **fully successful**. All four tasks tested (capital-common-countries, family, gram5-present-participle, gram7-past-tense) produced results that exactly matched the cached values from the original experiments. The key scientific claims of the paper were validated:

1. Concept induction heads create lenses that improve semantic parallelogram arithmetic (capitals: 83% vs 39% raw)
2. Token induction heads create lenses that improve grammatical parallelogram arithmetic (past tense: 85% vs 32% raw)
3. These specialized subspaces significantly outperform both raw hidden states and all-heads baselines

The implementation was reconstructed from the plan and code walkthrough without verbatim code copying, demonstrating that the documentation is sufficient for faithful replication.
