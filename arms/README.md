# The two reproduced arms

The scaling study measures seven RAG paradigms against corpus size. Two of them are
re-measured here on our own ladder, under the study's reader and its shared settings, so
that a published curve and a measured one sit on the same axes:

| arm | what it is | where it runs |
|---|---|---|
| 1 — BM25 | top-5 chunks, read by the study's reader | this host |
| 2 — File-System Agent | shell exploration of the tier tree, 80 LLM calls/question | this host |

A third arm — the same BM25 retrieval read by an Aethos-tier model — runs elsewhere and
consumes a file this one writes. A fourth is Aethos itself. Neither is built here.

## The settings, and why they are constants

`arms/common.py` holds them, and nothing takes them from the environment:

| setting | value | |
|---|---|---|
| retrieval depth | 5 chunks | `TOP_K` |
| chunking | 1,200 tokens, 100 overlap | `ladder.common.CHUNK_SIZE`/`CHUNK_OVERLAP` |
| agent budget | 80 LLM calls/question | `MAX_LLM_CALLS` |
| reader | `Qwen/Qwen3.6-27B` via vLLM | `READER_MODEL` |
| decoding | temperature 0, top-p 1.0, thinking off | `src/llm/vllm_llm.py` |

The settings *are* the reproduction. A knob that can drift between tiers would make the
rungs incomparable to each other, and nothing in the resulting curve would show it — so
the ladder's four rungs are measured by one configuration or the comparison is void.
Passing `--top-k` or `--max-llm-calls` anything else prints a warning saying the run is
not comparable to the published curve; it is there for smoke tests, not for tuning.

Chunk size and the document text come from `ladder.common`, imported rather than
restated. A tier's published token and chunk counts were validated against the study's
Table 7 over exactly that string and that window, so a retriever chunking a
differently-joined document would be searching a corpus the ladder never measured.

## What this adds to the repository's own runners, and why

The phrase "configure the runners" undersells it. Three of the study's shared settings
had no knob to turn:

**BM25 was document-level.** `src/scripts/answer_generation/index_document_bm25.py`
indexes whole documents into one `text` field and its runner retrieves whole documents;
there is no chunker anywhere near it. The study retrieves top-5 *chunks*. Rather than
change that pair, `arms/index_bm25.py` and `arms/bm25.py` are a second, chunked index and
runner alongside them — because `ladder.pool` mines its trap and lure candidates from the
document-level index's top-200, so re-pointing it at chunks would change a bedrock phase
5 has already committed.

**The agent was wall-clock bounded.** `run_agent_conversation` took `timeout_seconds` and
no call ceiling, so `max_llm_calls` is new. The last call of the budget is reserved for
the forced finish, so a question that spends the whole budget still answers instead of
returning the empty string. Retries after a transport error and context-compaction calls
are not charged, since neither is a step the agent chose to take.

**And its clock is a second ceiling.** The shipped runner cuts a question off at 600
seconds, so an arm keeping it can report an 80-call budget while *time* did the cutting —
at a point that moves with how loaded the box is. Measured at T0: a mean of 37 calls at
roughly 4.3s each, so 80 calls is around six minutes of that pacing and the shipped clock
would not always have bound. But per-call latency rises as the conversation grows, and
that margin is thin enough to bind sometimes and silently. The wall clock is raised to an
hour and becomes a backstop; the run's report counts `budget_exhausted` and
`cut_off_by_clock` separately, and if the second is not near zero the arm did not run
under the budget it names.

**The reader was unreachable.** `get_llm()` spoke the OpenAI Responses API against
`api.openai.com` with no `base_url`, and vLLM serves Chat Completions. `src/llm/vllm_llm.py`
is a third provider, selected by `LLM_PROVIDER=vllm`.

## Two things the arms do not see

**The organizational pages.** A tier is its manifest's lines *plus two* — the company
overview and the initiative index, which are not corpus documents and sit outside
`sources/`. Neither reproduced arm reads them: BM25 indexes documents by id and they have
none, and the agent's working directory is `sources/`. This matches the study's own
runners, which are rooted the same way. It is nevertheless an asymmetry with the Aethos
arm, which ingests all 1,144, and it belongs in the methodology note.

Measured at T0: 1,142 documents, 1,688,257 tokens, **2,011 chunks**. The tier's
provenance records 2,016 chunks over 1,692,700 tokens — the difference is exactly those
two pages (4,443 tokens, 5 chunks). The chunker reproduces phase 5's count to the token
on everything it can see.

**Anything outside the rung.** Each tier is its own index and its own corpus root, and
the tier's `uuid_index` — built from the tier tree, never the corpus's — is what the
agent's `select_doc_by_dsid` validates against. The search space is the experiment's
independent variable, so an arm searching wider than the rung it reports would be
measuring nothing. `report_run` fails the run if any answer names a document the manifest
does not.

## Running a rung

```bash
# On a workstation: cut the tier out of the corpus (never point an arm at the corpus —
# it carries four dsid collisions, one of them gold, so every tier would fail).
python -m ladder.materialize --tier T0 --out /data/tier-T0

# On the baseline host:
export LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8000/v1
python -m arms.index_bm25 --tier-tree /data/tier-T0 --recreate
python -m arms.bm25   --tier-tree /data/tier-T0 --out-dir results/T0/bm25
python -m arms.agent  --tier-tree /data/tier-T0 --out-dir results/T0/agent
```

Both commands resume: re-run the same line and it picks up where it stopped. Both refuse
to call a short run complete — a missing row exits non-zero, because a short file scores
as a shorter suite rather than as a worse one.

An output directory is pinned to the settings that wrote it, in a `run.json` beside the
answers. Resume keys off the question id alone, so without that pin a directory would
absorb rows from two configurations — a `--top-k 1` run followed by a `--top-k 5` one
makes no calls and reports the first run's answers as top-5 — and every row would be
well-formed and the file the right length. A second run under different settings exits
non-zero naming what differs; give it its own directory.

`arms/bm25.py` writes two files. `answers.jsonl` is the official format the scorer reads.
`contexts.jsonl` is the control arm's input: the same questions, the same documents, and
**the context block this arm's reader was given, verbatim**. That arm must differ from
this one in the model and nothing else, so it reads that string rather than reassembling
one from chunks — which means the rendering in `format_context_chunks` is a published
interface, and changing its wording makes cells measured before and after incomparable.

### A failure is a wrong answer, never a missing row

The official metrics evaluation *skips* a row carrying neither an answer nor document
ids. So an unanswered question written as an empty row leaves the denominator, and
twenty of them would have 480 answered questions scored as the whole suite — with failure
concentrated on the hard ones, which raises the score. Both arms write a constant
sentinel answer instead, with the real error beside it in `failure`, so a failure scores
zero for correctness and completeness. The shipped agent runner writes the empty row;
these do not.

### Parallelism

Both default to one question at a time. Above that, the recorded per-question latency is
a queueing time rather than the arm's. Whatever value a tier used, every other tier has
to use, and the run's summary records it.

## The host

`arms/host/create-instance.sh` builds it; `arms/host/startup.sh` is its boot script,
idempotent because a Spot instance restarts. It serves the reader on `:8000`, the
embedding model on `:8001` and OpenSearch on `:9200`.

`Qwen/Qwen3-Embedding-0.6B` is stood up because it is part of the reader stack the study
serves, but **neither reproduced arm queries it** — BM25 is lexical and the agent greps.
The dense paradigms that would use it are not among the two being re-measured.

This host is not the measurement instance and must never become it. The Aethos arm is
timed single-stream on an idle client-spec deployment, and an index build or an agent
sweep beside it would be measuring the neighbour.

### When the reader will not start

Two failures seen standing this host up, both of which look like software and are not:

**`CUDA error: unspecified launch failure` on every worker.** Check `dmesg | grep -i xid`
first. An `Xid 74` (NVLink fatal) followed by `Xid 154` (GPU reset required) means the
card is faulted, not that vLLM is misconfigured — no flag fixes it, and `--enforce-eager`
is a red herring. Stop and start the instance; the weights are on the persistent boot
disk, so it comes back in minutes rather than re-downloading 52GB. A Spot instance can
land on a bad host, and this is what that looks like.

**`max_num_seqs (256) exceeds available Mamba cache blocks`.** The reader is a hybrid
Mamba/attention model, and every decode sequence needs its own Mamba cache block. vLLM's
default concurrency asks for more blocks than fit, and it refuses to capture CUDA graphs
rather than degrading — so the engine never starts. `startup.sh` caps `MAX_NUM_SEQS` well
below the limit; the arms ask one question at a time, so the cap costs them nothing.

**The embedding server will not start.** Two separate causes, both about the sliver of
card 0 the reader leaves it. `--task embed` was removed in vLLM 0.28 — it is `--runner
pooling` now. And vLLM refuses to start unless the KV cache can hold one sequence of the
model's full context window, which at this model's 32k default it cannot; `startup.sh`
caps `EMBEDDER_MAX_MODEL_LEN` well below that. Nothing the two reproduced arms do depends
on this server, so a failure here does not block a run.

**The driver will not build.** Do not use a stock Ubuntu image. NVIDIA's bundled
installer fails to compile its kernel module against recent GCP kernels
(`os-interface.h: No such file or directory` on 6.17.0-1022-gcp). The Deep Learning VM
image `create-instance.sh` selects ships the driver, and `startup.sh` only verifies it.

### Preflight

Before a multi-hour run, both arms probe the reader. Two properties are checked because
both fail into a plausible result rather than an error, and neither is visible in the
output file:

* **Thinking is off.** `chat_template_kwargs` is passed through to whatever chat template
  the server loaded, and a template that ignores the flag accepts it silently. What that
  produces is not a crash but reasoning text inside answers, which the official scorer
  grades as part of the answer. Checked at the top of a run rather than stripped per
  question, because stripping would be this harness editing the system under test.
* **Tool calls are served** (agent arm only). Without `--enable-auto-tool-choice`, or
  with a `--tool-call-parser` that does not match the model's template, the model returns
  prose *describing* a call. The loop sees no tool call, nudges, and the agent spends all
  80 calls talking — 500 confidently wrong answers, hours to discover.
