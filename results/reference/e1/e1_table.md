Protocol: 500 corpus docs, 100 queries, GoodWiki-Long test, batch 8 (Jina-v3 at batch 4 — batch 8 exceeds 24 GB on this card), 1 warmup + 3 timed repeats, single RTX 4090 (24 GB), otherwise-idle GPU.

| System                     | Index (s) |  ms/query | Peak GPU (GB) | Index (MB) |
|----------------------------|-----------|-----------|---------------|------------|
| *Sparse lexical* | | | | |
| BM25                       |       0.2 |       6.5 | -- |        2.6 |
| TF-IDF                     |       0.1 |       1.1 | -- |        3.4 |
| *Native long-context dense (truncate 8K)* | | | | |
| BGE-M3                     |      37.6 |     430.3 |          9.80 |        2.0 |
| Jina-Embeddings-v3         |      26.0 |     336.7 |         18.90 |        2.0 |
| Stella-en-1.5B-v5          |      30.6 |     318.9 |         14.34 |        3.1 |
| Nomic-Embed-v1.5           |      13.4 |     160.5 |          4.81 |        1.5 |
| *Bare GN (chunked mean-pool, 512)* | | | | |
| GTE-small (chunked)        |       4.1 |      23.0 |          0.23 |        0.8 |
| GTE-base (chunked)         |       7.7 |      42.7 |          0.61 |        1.5 |
| GTE-large (chunked)        |      21.6 |     121.4 |          1.56 |        2.0 |
| BGE-base (chunked)         |       7.7 |      42.9 |          0.61 |        1.5 |
| BGE-large (chunked)        |      21.6 |     121.4 |          1.56 |        2.0 |
| *REIGN, uncached GN (cold cache)* | | | | |
| REIGN + GTE-small          |       3.3 |      19.8 |          0.35 |        1.5 |
| REIGN + GTE-base           |       6.6 |      39.7 |          0.76 |        1.5 |
| REIGN + GTE-large          |      19.9 |     118.2 |          1.73 |        1.5 |
| *REIGN, cached GN embeddings (warm)* | | | | |
| REIGN + GTE-small          |       0.2 |       0.4 |          0.24 |        1.5 |
| REIGN + GTE-base           |       0.2 |       0.5 |          0.55 |        1.5 |
| REIGN + GTE-large          |       0.2 |       0.5 |          1.45 |        1.5 |

**One-time GN cache build** (the cost being amortised):

| System | Build time | Per document | Cache size |
|---|---:|---:|---:|
| REIGN + gte-base     |     10.8 s |     18.0 ms/doc |      9.2 MB |
| REIGN + gte-large    |     32.0 s |     53.3 ms/doc |     11.2 MB |
| REIGN + gte-small    |      5.4 s |      8.9 ms/doc |      6.2 MB |

**Uncached REIGN vs its own chunked GN**, **and the cached speed-up**:

| GN | GN chunked (ms/q) | REIGN uncached (ms/q) | ratio | REIGN cached (ms/q) | cached speed-up |
|---|---:|---:|---:|---:|---:|
| gte-small  |     23.0 |     19.8 |   0.86x |      0.4 |   49.3x |
| gte-base   |     42.7 |     39.7 |   0.93x |      0.5 |   85.1x |
| gte-large  |    121.4 |    118.2 |   0.97x |      0.5 |  229.4x |

## Reading note

`REIGN uncached` measures at 0.86-0.97x its own chunked-GN baseline, i.e. slightly
*faster*, which is counter-intuitive because REIGN runs the same GN forward plus a
small encoder on top. It is not doing less work: tokenising 40 GoodWiki-Long test
documents through both paths yields **167 chunks each, identical for 40/40
documents**, so the GN forward pass is the same in both.

The gap is a batching artifact of the baseline implementation. `DenseEncoder`'s
`chunk_pool` path encodes one document at a time (`for text in texts`), batching
only that document's chunks, so it issues many small partially-filled GPU batches.
`ReignFeatureExtractor` tokenises a whole batch of documents at once and encodes all
their chunks together, filling the GPU better.

**Therefore this should be read as parity, not as a speed-up.** The defensible
claim is that the cross-chunk encoder adds negligible cost on top of the GN
forward: over k~16 chunk embeddings its cost disappears into measurement noise
against the GN itself. Any wording stronger than parity would be an artifact of
how the baseline batches, not a property of REIGN.

The cached rows are a different and genuine claim: 49-229x lower per-query latency,
because the GN forward is replaced by an HDF5 read. Note the cached peak-memory
figures still include the GN weights resident on the device (the encoder object
keeps it loaded to serve fresh queries); a corpus-only indexing deployment that
never re-encodes text would not need it.
