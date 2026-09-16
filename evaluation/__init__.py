"""
Evaluation of the retrieval + synthesis system.

Run everything from the repository root as modules:

    python3 -m evaluation.sample_gold       draw gold chunks → query CSV skeleton
    python3 -m evaluation.validate_queries  check a query file against the index
    python3 -m evaluation.run_eval          drive /api/chat over Sets A and B
    python3 -m evaluation.forced_pairing    Set C: synthesis over irrelevant passages
    python3 -m evaluation.bm25_baseline     lower bound, offline, no LLM
    python3 -m evaluation.frontier_baseline upper bound, whole corpus in a cached prefix
    python3 -m evaluation.report            aggregate a run directory into a report

See evaluation/README.md for the design and the order to run them in.
"""
