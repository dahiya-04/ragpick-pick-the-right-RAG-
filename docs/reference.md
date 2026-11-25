# Reference

## What a run produces

Everything goes to `results/<run_name>_<date-time>/`:

| File | Contents |
|---|---|
| `report.html`, `report.md` | The full report: headline, recommendation, charts, tables, failure examples |
| `summary.csv` | One row per setup with every score, range, verdict, time and cost |
| `by_type.csv` | Every setup's score per question type |
| `failures.csv` | Every wrong answer of the best setup, with its likely cause and an empty column for your own note |
| `parse_report.csv` | Per document: pages, tables, removed headers, circular number, date, replaced-by |
| `evidence_findability.csv` | Whether each evidence sentence still exists after cutting, per chunker |
| `per_question.jsonl` | Every question x setup with all details, for your own analysis |
| `charts/` | Accuracy, Recall@k and question-type charts |

## Configuration reference

A short `config.yaml`:

```yaml
run_name: rbi
corpus:
  path: data/corpus
testset:
  path: testset/questions.jsonl
  split: all                     # all | dev | test

embedders:
  azure-small: {type: azure, api: openai, deployment: text-embedding-3-small}

chunkers:
  - {name: fixed, type: fixed, size: 400, overlap: 50}
  - {name: structure, type: structure, max_size: 500, min_size: 40}

retrievers:
  - {name: bm25, type: bm25}
  - {name: dense, type: dense, embedder: azure-small}
  - {name: hybrid, type: hybrid, embedder: azure-small}

evaluation:
  k: [1, 3, 5]
  context_k: 5                   # how many pieces the answer model reads
  baseline: {chunker: fixed, retriever: dense, reranker: none}

generation:
  llm: {type: goedel, model: deepseek/deepseek-v4.1-flash}
judge:
  llm: {type: goedel, model: deepseek/deepseek-v4.1-flash}

decision:
  max_latency_ms: 3000           # optional limit on time per answer
  max_usd_per_1k:                # optional limit on cost per 1,000 answers
  prefer: cost                   # cost | speed

prices:                          # USD per million tokens [input, output]
  deepseek/deepseek-v4.1-flash: [0.10, 0.40]
```

### Available parts

| Part | Options |
|---|---|
| Documents | PDF (with tables), TXT, MD, HTML. Removes repeated headers and footers and Hindi lines. Reads RBI circular numbers, dates and which rules replace which. |
| Chunkers | `fixed`, `recursive`, `structure`, `semantic` |
| Embedders | `azure` (Azure AI Foundry), `sentence-transformers` (local), `hashing` (offline stand-in) |
| Retrievers | `bm25`, `dense`, `hybrid`, plus `superseded_filter: [false, true]` to also test hiding old rules |
| Rerankers | `cross-encoder`, `lexical`, `none` |
| Answer and judge models | `goedel`, `openai-compatible` (any OpenAI-style server), `anthropic` (Claude), `extractive` and `lexical` (offline stand-ins) |

To add your own part, write a class and set `type: my_module:MyClass`.

To save model calls on large grids, `generation.configs: top:4` runs search for every setup but writes answers only for the 4 setups with the best search scores.

## Project layout

```
ragpick/
  cli.py          commands (init, demo, draft, accept, split, parse, run, judge-sample, agreement)
  parsing.py      read documents, clean text, extract metadata, detect replaced rules
  chunking.py     fixed, recursive, structure, semantic chunkers
  embeddings.py   Azure, sentence-transformers, hashing; embedding caches
  retrieval.py    BM25, dense, hybrid search
  rerank.py       rerankers
  llm.py          model clients (OpenAI-compatible, Claude) with a reply cache
  generation.py   the fixed answer instructions
  judge.py        grading rules and the judge
  metrics.py      Recall@k and MRR with evidence-sentence matching
  stats.py        95% ranges and tie tests
  decide.py       the final recommendation
  drafting.py     question drafting and its checks
  pipeline.py     runs everything in order
  report.py       tables, charts, report.md and report.html
examples/rbi-synthetic/   the experiment in section 7 (synthetic documents + 36 questions)
tests/                    24 offline tests: pytest
docs/images/              charts used in this README
```
