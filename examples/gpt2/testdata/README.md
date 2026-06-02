# gpt2 testdata

The reference-fixture tests in `bpe_test.go` verify the Go BPE implementation
against ground-truth ids that were computed at dispatch time using the
canonical GPT-2 vocab.json and merges.txt from HuggingFace
(`https://huggingface.co/gpt2/resolve/main/`).

To keep the repo small and CI strictly offline, the **vocab.json
(~1.0 MB) and merges.txt (~445 KB) files are NOT committed**. The tests
expect them at:

```
examples/gpt2/testdata/vocab.json
examples/gpt2/testdata/merges.txt
```

If both files are absent, the reference-fixture tests `t.Skip` and only the
unit-level mechanics tests (tiny hand-crafted vocab, byte-bijection round
trip, pre-tokenizer regex) run. CI exercises the unit tests only.

To run the full reference suite locally, fetch the canonical files:

```bash
curl -sLfo examples/gpt2/testdata/vocab.json \
  https://huggingface.co/gpt2/resolve/main/vocab.json
curl -sLfo examples/gpt2/testdata/merges.txt \
  https://huggingface.co/gpt2/resolve/main/merges.txt
```

For integrity, the tests verify the SHA-256 of each file before using it:

```
vocab.json:  196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
merges.txt:  1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
```

A mismatch indicates HuggingFace shipped a new version; regenerate the
embedded fixture ids in `bpe_test.go` using `python3 testdata/gen_fixtures.py`
(see the dispatch notes for the canonical Python implementation).
