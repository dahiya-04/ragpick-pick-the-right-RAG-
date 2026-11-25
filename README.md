# ragpick

**One command runs a full RAG experiment on your own documents.** It compares chunking strategies, retrievers, rerankers and filters, and writes every table, chart and failure analysis you need to answer *"which RAG setup actually works on my documents, and how do I know?"*

```bash
ragpick run config.yaml
```

## What one run produces

`results/<run_name>_<timestamp>/`

| File | What it is |
|---|---|
| `report.md` / `report.html` | One-line headline finding, main table, chart, insights by question type, failure examples. `report.md` is ready to paste into a README. |
| `summary.csv` / `summary.md` | One row per setup: Recall@k, MRR, Hit-all@k, accuracy, faithfulness, abstention, latency (p50), $ per 1k queries |
| `by_type.csv` / `by_type.md` | Every setup broken down by question type (lookup, numeric, multi_doc, latest_rule, unanswerable …) |
| `evidence_findability.csv` | Whether the gold passage still exists in *any* chunk after parsing and chunking, per chunker |
| `failures.csv` | Every wrong answer from the best setup, with an automatic diagnosis and a blank `human_diagnosis` column |
| `parse_report.csv` | Per document: pages, tables found, headers removed, Hindi lines, likely-scanned flag, circular no., date, superseded-by |
| `charts/*.png` | Recall@k by setup, accuracy by setup, accuracy-by-question-type heatmap |
| `per_question.jsonl` | Every question × setup with all scores (for your own notebooks) |

Automatic diagnoses: `parsing_or_chunking_lost_evidence`, `retrieval_miss`, `retrieval_miss_superseded_retrieved`, `partial_retrieval`, `false_abstain`, `unfaithful_generation`, `generation_followed_superseded_rule`, `generation_error`, `answered_unanswerable`.

## Install

```bash
pip install -e .            # core: parsing, BM25, metrics, reports
pip install -e ".[all]"     # + sentence-transformers (dense, semantic chunking, reranker) + anthropic (generation, judge)
```

## Try it first (offline, free, about 10 seconds)

```bash
ragpick demo
```

This runs a synthetic sample corpus of RBI-style circulars (**not real RBI text**) through 4 chunkers × 3 retrievers × 2 rerankers × 2 superseded filters = 48 setups. It uses stand-in components (hashing embedder, extractive "generator", lexical judge), so it needs no API key and no model download.

## Your own project

```bash
ragpick init my-project                      # config.yaml, testset template, folders
# 1. put PDFs in my-project/data/corpus/
ragpick parse my-project/config.yaml         # check parse_report.csv before labelling anything
# 2. write 100-150 questions in my-project/testset/questions.jsonl, or let the LLM draft them:
ragpick draft my-project/config.yaml --n 60          # -> testset/draft_questions.csv
#    review the CSV: keep=n to reject, fix wording; then
ragpick accept my-project/testset/draft_questions.csv   # appends kept rows to questions.jsonl
ragpick split my-project/testset/questions.jsonl     # stratified 30% dev / 70% test
ragpick run my-project/config.yaml --split dev --limit 10 --set generation.enabled=false   # quick smoke test
ragpick run my-project/config.yaml --dry-run         # show grid + number of LLM calls
ragpick run my-project/config.yaml                   # the real thing (test split)
```

Copy `.env.example` to `.env` and fill in `AZURE_EMBEDDING_ENDPOINT`, `AZURE_EMBEDDING_API_KEY` (embeddings) and `GOEDEL_API_KEY` (generation and judging). Every LLM call is cached in `.ragpick_cache/llm_cache.sqlite`, so re-running a report is free. To limit cost, use `generation.configs: top:4`: retrieval runs for every setup, but only the 4 best setups by Recall@k get generated answers.

### Test-set format (one JSON object per line)

```json
{"id": "q017", "type": "latest_rule", "question": "Within how many days must a card closure request be honoured?",
 "answer": "Seven working days.",
 "evidence": [{"text": "Any request for closure of a credit card shall be honoured within seven working days ...", "doc": "credit_card_md_2022"}]}
```

- `evidence` is the **text** of the gold passage, not a chunk ID. Different chunkers cut documents differently, so a chunk counts as a hit when it covers ≥50% of the gold passage's word 3-grams (or is ≥80% made of gold text). See `ragpick/metrics.py`.
- `doc` (optional) restricts the match to one file.
- Multi-document questions list several evidence passages. Recall@k is the share found; Hit-all@k means all were found.
- `type: unanswerable` with empty evidence. The answer is correct only if the system replies `NOT FOUND`.

### How the final recommendation is made

1. **How sure are we?** Each score gets a 95% range: the questions are re-sampled 2,000 times. Every setup is compared with the best one on the same questions, using a paired sign-flip test (the exact McNemar test for right/wrong scores). If the gap could come from luck more than 5% of the time, they are **tied**. The report also says roughly how many questions it would take to separate them.
2. **Your limits:** setups that break `decision.max_latency_ms` (p95 end-to-end) or `decision.max_usd_per_1k` are excluded.
3. **The choice:** among the best remaining setup and everything tied with it, the report recommends the cheapest one (`prefer: cost`) or the fastest one (`prefer: speed`). Differences under 10% count as equal, and the higher-quality setup wins.

End-to-end time = search + a live-measured query-embedding call + answer generation. Cost is in dollars when `prices` lists your models. Otherwise it is LLM tokens per query, which still differs between setups because chunk size changes the prompt length.

### Drafted questions

`ragpick draft` uses the generation LLM to write a mix of lookup, numeric, reworded, multi-document and unanswerable questions, spread across all documents. Before you see them, each draft is checked automatically:

- its evidence must appear word for word in the source passage;
- near-duplicates are dropped;
- each unanswerable question is searched for in the corpus and dropped if a passage answers it.

The `notes` column flags drafts worth a closer look: ones from superseded documents, and ones that copy the passage wording (easy for keyword search). Drafted questions are easier than hand-written ones, so add a handful of your own hard ones (latest-rule, tricky wording). Accepted rows are tagged `"source": "drafted"`.

### Checking the judge yourself

```bash
ragpick judge-sample results/<run>  --n 50    # human_grading.csv, judge labels hidden
# fill human_correct / human_faithful with y/n
ragpick agreement results/<run>               # agreement % + Cohen's kappa, appended to report.md
```

## How the comparison stays fair

- One fixed generation prompt and one LLM for every setup (`ragpick/generation.py`). Only retrieval changes.
- Dev/test split: tune on `--split dev`, and report `test` once at the end.
- Deterministic rules grade abstentions before the LLM judge sees anything. Faithfulness is judged against the retrieved passages only.
- Latency is measured per query for retrieval and reranking, and separately for generation.

## Components

| Stage | Built-in `type`s |
|---|---|
| Parsing | PDF (PyMuPDF, tables → Markdown), TXT, MD, HTML; header/footer removal; Hindi-line removal; RBI metadata (circular no., ref no., date, department, supersession) + CSV overrides |
| Chunking | `fixed`, `recursive`, `structure` (numbered paras / chapters / annexes), `semantic` |
| Embedders | `sentence-transformers` (bge-m3, e5 …), `azure` (Azure AI Foundry: text-embedding-3-*, Cohere-embed-v3-*, embed-v-4-0), `hashing` (offline stand-in) |
| Retrieval | `bm25`, `dense`, `hybrid` (RRF) + optional superseded-document filter |
| Reranking | `cross-encoder` (bge-reranker …), `lexical` (toy), `none` |
| Generation | `goedel` (Goedel Machines, DeepSeek V4.1 Flash), `openai-compatible` (any OpenAI-style server), `anthropic` (Claude), `extractive` (no-LLM floor) |
| Judge | `goedel` / `openai-compatible` / `anthropic`, `lexical` (heuristic) |

Add your own component by setting `type: my_module:MyClass` in the config (see `ragpick/registry.py`).

The default LLM for answers and judging is `deepseek/deepseek-v4.1-flash` on Goedel Machines (OpenAI-compatible, reasoning off by default). Because no structured-output parameter is documented for it, the judge describes the JSON fields in its prompt and parses the first JSON object in the reply. With `type: anthropic`, `claude-opus-5-5` is used, with server-side refusal fallbacks enabled (`fallbacks: default`); set `fallbacks: null` to turn that off.
