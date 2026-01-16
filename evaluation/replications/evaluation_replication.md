# Replication Evaluation: Vector Arithmetic in Concept and Token Subspaces

## Overview

This document evaluates the replication of "Vector Arithmetic in Concept and Token Subspaces" using the binary checklist criteria.

## Replication Process

### What Was Replicated
1. **Capital Cities Task**: Parallelogram arithmetic (Athens - Greece + China = Beijing)
2. **Family Relations Task**: Semantic analogy task
3. **Present Participle Task**: Grammatical transformation task
4. **Past Tense Task**: Grammatical transformation task
5. **Layer Sweep Analysis**: Performance across layers [0, 4, 8, 12, 16, 20, 24, 28, 31]

### Implementation Approach
- Reimplemented core functions from understanding of plan.md and CodeWalkthrough.md
- Used nnsight framework for model tracing (as specified in original code)
- Built OV sum matrices by loading pre-computed causal scores
- Computed word representations and evaluated nearest neighbor accuracy

---

## Replication Evaluation - Binary Checklist

### RP1. Implementation Reconstructability

**Status: PASS**

**Rationale**: The experiment could be fully reconstructed from the plan.md and CodeWalkthrough.md without missing steps:
- The plan clearly describes the methodology: building OV lenses from top-k concept/token heads
- The code walkthrough explains the script structure and execution flow
- Pre-computed causal scores are provided in the cache directory
- Data files are well-organized with clear format (space-separated word tuples)
- No major guesswork or unclear dependencies were encountered

The only minor ambiguity was the exact file paths for cached scores, which was easily resolved by exploring the cache directory structure.

---

### RP2. Environment Reproducibility

**Status: PASS**

**Rationale**: The environment was successfully restored and executed:
- Model: `meta-llama/Llama-2-7b-hf` loaded via nnsight
- All dependencies (torch, nnsight, matplotlib, json) were available
- Pre-computed causal scores were present in `cache/causal_scores/Llama-2-7b-hf/`
- Data files were present in `data/word2vec/`
- GPU (NVIDIA A40) was available and used as specified
- No version conflicts or missing dependencies encountered

---

### RP3. Determinism and Stability

**Status: PASS**

**Rationale**: Results are fully deterministic and reproducible:
- All numerical results **exactly match** the cached values from the original repository
- The operations (matrix multiplication, cosine similarity, nearest neighbor search) are deterministic
- No random seeds required as there is no stochastic component in the evaluation
- Zero variance observed between replication and cached results across:
  - All 4 methods (raw, concept, token, all)
  - All 9 layers tested
  - All 4 tasks replicated

Example verification (Capital Cities, Layer 20):
- Raw: Replicated=15.81%, Cached=15.81% (exact match)
- Concept: Replicated=89.53%, Cached=89.53% (exact match)
- Token: Replicated=7.31%, Cached=7.31% (exact match)
- All: Replicated=18.18%, Cached=18.18% (exact match)

---

### RP4. Demo Presentation

**Status: NA**

**Rationale**: This repository is not classified as demo-only. The full experimental pipeline is provided and replicable:
- Complete source code in `scripts/` directory
- Full dataset in `data/` directory
- Pre-computed intermediate results in `cache/` directory
- The replication was performed on the full experiments, not a demo subset

---

## Ambiguities and Issues Encountered

### Minor Issues
1. **Initial GPU Memory**: Another process was using GPU memory, requiring use of `device_map='auto'` instead of `device_map='cuda'`. This did not affect results.

### No Major Issues
- All file paths were discoverable through directory exploration
- All data formats were consistent with documentation
- No missing dependencies or version conflicts

---

## Summary

| Criterion | Status |
|-----------|--------|
| RP1. Implementation Reconstructability | PASS |
| RP2. Environment Reproducibility | PASS |
| RP3. Determinism and Stability | PASS |
| RP4. Demo Presentation | NA |

**Overall Assessment**: The replication was **fully successful**. All experimental results exactly match the original cached values, demonstrating perfect reproducibility. The plan.md and CodeWalkthrough.md provided sufficient information to reconstruct the experiments without ambiguity. The repository is well-organized with clear data/code separation and pre-computed resources that enable faithful replication.
