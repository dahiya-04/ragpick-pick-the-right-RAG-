# Architecture: who talks to whom

ragpick is one program that talks to up to three AI models, each with one job.

| Role | Job | Default model |
|---|---|---|
| **Embedding model** | Turns text into numbers for meaning-based search | Azure AI Foundry `text-embedding-3-small` |
| **Answer model** | Writes the answer from the found chunks | Goedel Machines `deepseek/deepseek-v4.1-flash` |
| **Judge model** | Grades each answer: correct? faithful? | Goedel Machines `deepseek/deepseek-v4.1-flash` |
| **Question writer** | Drafts test questions (step 1) | the answer model |

Any of these can be swapped in `config.yaml`: Claude, any OpenAI-compatible server, local sentence-transformers, or your own class.

## Components

```mermaid
flowchart TB
    subgraph you["You provide"]
        DOCS[("Documents<br/>PDF, TXT, HTML")]
        QS[("Test questions")]
        CFG["config.yaml + .env keys"]
    end

    subgraph ragpick["ragpick"]
        PARSE["Parser<br/>text, tables, dates,<br/>circular numbers, which rules<br/>replace which"]
        CHUNK["Chunkers<br/>fixed, structure,<br/>recursive, semantic"]
        SEARCH["Retrievers<br/>BM25, dense, hybrid<br/>+ optional reranker"]
        SCORE["Search scoring<br/>Recall@k, MRR"]
        GEN["Answer step"]
        JUDGE["Grading step<br/>rules first, then judge"]
        STATS["Confidence check<br/>95% ranges, ties"]
        DECIDE["Decision<br/>limits, cost, speed"]
        REPORT["Report writer"]
        CACHE[("Cache<br/>every model reply and<br/>embedding is saved")]
    end

    subgraph models["External AI models"]
        EMB["Embedding model"]
        LLM["Answer model"]
        JLLM["Judge model"]
    end

    DOCS --> PARSE --> CHUNK --> SEARCH
    QS --> SEARCH
    CFG -.-> ragpick
    SEARCH <--> EMB
    SEARCH --> SCORE
    SEARCH --> GEN
    GEN <--> LLM
    GEN --> JUDGE
    JUDGE <--> JLLM
    SCORE --> STATS
    JUDGE --> STATS --> DECIDE --> REPORT
    EMB -.-> CACHE
    LLM -.-> CACHE
    JLLM -.-> CACHE
```

## One question, step by step

This is what happens for a single question in a single setup:

```mermaid
sequenceDiagram
    autonumber
    participant P as ragpick
    participant C as Cache
    participant E as Embedding model
    participant A as Answer model
    participant J as Judge model

    P->>C: Do we already have this question's embedding?
    alt not saved yet
        P->>E: Embed the question
        E-->>P: List of numbers
        P->>C: Save it
    end
    P->>P: Search chunks, keep the top 3, check if the gold chunk is among them
    P->>C: Do we already have an answer for these exact chunks?
    alt not saved yet
        P->>A: Question + 3 chunks + fixed instructions
        A-->>P: Answer with [1] [2] citations, or "NOT FOUND"
        P->>C: Save it
    end
    alt answer is "NOT FOUND"
        P->>P: Grade by rule (no judge needed)
    else normal answer
        P->>J: Question + correct answer + chunks + system answer
        J-->>P: correct: yes/no, faithful: yes/no, reason
    end
```

**The cache matters.** Every model reply is saved on disk. Running the same experiment again costs nothing, and adding new setups only pays for the new work.
