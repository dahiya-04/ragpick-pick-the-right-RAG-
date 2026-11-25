# Limitations

- **Small question sets give wide ranges.** With 36 questions, a 3-point difference cannot be proven. Use 100 or more questions for decisions that matter. The report tells you roughly how many you need.
- **The judge is an AI model.** It can be wrong. Grade 50 answers by hand with `judge-sample` and check `agreement`. Using a different model for judging than for answering is safer.
- **Drafted questions are easier than real ones.** Add some hard questions by hand: old-versus-new rules, trick wording, codes.
- **The example documents are synthetic.** Real documents bring scanned pages, complex tables and messy layouts. Check `parse_report.csv` first.
- **Speed numbers depend on the network and the provider's load** at the time of the run.
- **Only the search part is compared.** The answer model and its instructions are fixed during a run. To compare answer models, run twice with different `generation.llm` settings.
