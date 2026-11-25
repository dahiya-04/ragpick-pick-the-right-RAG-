# How ragpick works: the four steps

```mermaid
flowchart LR
    S1["<b>Step 1: Draft</b><br/>AI writes test questions<br/>from your documents"] --> S2["<b>Step 2: Review</b><br/>You keep, fix<br/>or reject them"]
    S2 --> S3["<b>Step 3: Run</b><br/>Every setup answers<br/>every question; a judge<br/>grades each answer"]
    S3 --> S4["<b>Step 4: Decide</b><br/>Confidence check, then<br/>pick the best setup<br/>within your limits"]
```

## Step 1: Draft the test questions

To test a system you need questions with known correct answers. Writing 100 of them by hand takes hours, so `ragpick draft` asks an AI model to write them from your documents. It writes a mix of question types:

| Type | Example |
|---|---|
| lookup | "Which documents count as an Officially Valid Document?" |
| numeric | "Within how many days must NBFCs report frauds?" |
| paraphrase | Same fact, asked in everyday words that do not match the document's wording |
| multi_doc | Needs facts from two documents, for example "how did the rule change from 2017 to 2021?" |
| unanswerable | A believable question that the documents do **not** answer. The correct reply is "NOT FOUND". |

Every draft is checked automatically **before** you see it:

```mermaid
flowchart TD
    D["Draft question"] --> C1{"Is the supporting sentence<br/>really in the document,<br/>word for word?"}
    C1 -- no --> X["Thrown away"]
    C1 -- yes --> C2{"Does it make sense on its own?<br/>(no 'the passage', no 'the two documents')"}
    C2 -- no --> X
    C2 -- yes --> C3{"Is it a near-copy of<br/>a question we already have?"}
    C3 -- yes --> X
    C3 -- no --> C4{"Unanswerable type only:<br/>does any document<br/>answer it after all?"}
    C4 -- yes --> X
    C4 -- no --> K["Kept, with notes for you<br/>(for example 'vague: says these Directions')"]
```

## Step 2: Review

You open `draft_questions.csv` in Excel, set `keep` to `n` for bad rows, fix any wording, and run `ragpick accept`. Only your approved questions enter the test set.

## Step 3: Run

ragpick builds every setup you listed, asks each one every question, and grades every answer.

```mermaid
flowchart LR
    subgraph grid["Every combination is tested"]
        direction TB
        C1["fixed chunks"] --- R1["BM25"]
        C1 --- R2["dense"]
        C1 --- R3["hybrid"]
        C2["structure chunks"] --- R4["BM25"]
        C2 --- R5["dense"]
        C2 --- R6["hybrid"]
    end
    grid --> G["Same answer model,<br/>same instructions<br/>for every setup"]
    G --> J["Judge grades<br/>every answer"]
```

Only the search part changes between setups. The answer model and its instructions stay exactly the same, so any difference in results comes from the search.

## Step 4: Decide

```mermaid
flowchart TD
    A["All setups with their scores"] --> B{"Inside your limits?<br/>(time per answer, cost)"}
    B -- no --> E["Excluded"]
    B -- yes --> C["Find the best score"]
    C --> D["Collect every setup that is<br/><b>statistically tied</b> with the best"]
    D --> F["From the tied group, pick the<br/><b>cheapest</b> (or fastest)"]
    F --> G["Recommendation, with the reason"]
```

The idea: if two setups are equally good, choose the one that is cheaper and faster. Differences smaller than 10% count as equal, and then the higher score wins.
