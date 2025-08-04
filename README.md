# Cenliea


**CENLIEA: Cenliea: A Cross-Encoder NLI Framework Enhanced by LLM-Based Reasoning for Transferable, Domain-Agnostic Entity Alignment**

This repository contains all components for the CENLIEA and CENLIEA+ pipelines, combining deep natural language inference (NLI) and structured prompt engineering to align entities across heterogeneous knowledge graphs.

---

## 📌 Overview

CENLIEA introduces a neurosymbolic approach to entity alignment, where:
- Entity pairs are described using structured prompts.
- Bidirectional NLI inference generates vector embeddings.
- A lightweight classifier detects alignment based on semantic similarity.
- CENLIEA+ optionally incorporates LLM-generated similarity reasoning to improve robustness on hard cases.

This repository includes:
- ✅ Structured input preparation scripts (JSON/Parquet)
