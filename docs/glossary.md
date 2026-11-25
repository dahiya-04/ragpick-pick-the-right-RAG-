# Words you will see

| Word | Simple meaning |
|---|---|
| **Chunk** | A small piece of a document. Documents are cut into chunks so the search can find just the right part. |
| **Chunker** | The rule for cutting. `fixed` cuts every N words. `structure` cuts at the document's own numbered paragraphs (1.1, 1.2, ...). |
| **Retriever** | The search method that finds chunks for a question. |
| **BM25** | Search by matching **words** (like a book index). Good at exact terms such as "RBI/2023-24/53". |
| **Dense** | Search by matching **meaning**, using an embedding model. Finds "keep the paperwork" when the text says "maintain records". |
| **Hybrid** | Both searches together, with their rankings combined. |
| **Embedding model** | An AI model that turns text into numbers, so that texts with similar meaning get similar numbers. |
| **Setup** | One combination, for example "structure chunker + hybrid search". |
| **Superseded** | A rule that has been replaced by a newer rule. Old rules must not be used for answers. |
| **Judge** | An AI model that grades each answer against the correct answer. |
| **Recall@3** | Out of 100 questions, how often was the right chunk among the top 3 search results? |
| **Accuracy** | How many answers were correct. |
| **Faithfulness** | Did the answer only say things that the found chunks actually support (no made-up facts)? |
| **95% range** | The score could be a bit higher or lower with different questions. The true score is very likely inside this range. |
| **Tied** | Two setups are so close that the difference could be luck. |
