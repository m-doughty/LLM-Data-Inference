[![Actions Status](https://github.com/m-doughty/LLM-Data-Inference/actions/workflows/test.yml/badge.svg)](https://github.com/m-doughty/LLM-Data-Inference/actions)

NAME
====

LLM::Data::Inference - Structured LLM task layer with model-fallback, JSON parsing, and query-based routing

SYNOPSIS
========

```raku
use LLM::Data::Inference;

# Simple blocking LLM call (single backend — legacy shape)
my $task = LLM::Data::Inference::Task.new(
    :backend($my-backend),
    :system-prompt('You are a helpful assistant.'),
    :user-prompt('What is 2+2?'),
);
say $task.execute;  # "4"

# Model-fallback chain: try backends in order, advance to the next
# on model-specific failures (timeout, malformed output, 429, etc.),
# retry the head on transient errors (connection drop, 5xx), abort
# immediately on config errors (401 / 402 / 403).
my $task = LLM::Data::Inference::Task.new(
    :backends($primary, $secondary, $cheap-fallback),
    :user-prompt('Write a scene.'),
);
say $task.execute;  # serves from whichever backend produced a good response

# JSON output with schema checks — parser failures advance through
# the chain rather than retrying the same model on malformed JSON.
my $json-task = LLM::Data::Inference::JSONTask.new(
    :backends($primary, $fallback),
    :user-prompt('Return a JSON object with name and age.'),
    :required-keys('name', 'age'),
);
my %result = $json-task.execute;
say %result<name>;  # "Alice"

# Template-based prompts
my $pb = LLM::Data::Inference::PromptBuilder.new(
    :template('Write a {{genre}} story about {{topic}}.')
);
say $pb.render(%(:genre('fantasy'), :topic('dragons')));

# Query-based routing (orthogonal to the fallback chain — routers
# pick a backend, Tasks run fallbacks against that backend).
my $router = LLM::Data::Inference::Router.new(:default-backend($cloud-api));
$router.add-route('confidential OR restricted', $local-model);
$router.add-route('genre:technical', $reasoning-model);

my $backend = $router.select-backend($tags, $doc-id);
```

DESCRIPTION
===========

LLM::Data::Inference provides a structured task layer on top of LLM::Chat for use in data generation pipelines. It wraps the async LLM::Chat API into blocking calls with a three-bucket retry + model-fallback policy, JSON extraction, and content-based model routing.

LLM::Data::Inference::Task
--------------------------

Blocking LLM call with a model-fallback chain. Accepts either the legacy single `:$backend` or an ordered `:@backends` list; internally both are stored as a chain, and a single-backend chain behaves exactly like the pre-fallback Task on retry-same errors.

```raku
my $task = LLM::Data::Inference::Task.new(
    :backends($primary, $fallback),   # OR :backend($single)
    :system-prompt('Be helpful.'),    # Optional system prompt
    :user-prompt('Hello'),            # User message (required)
    :max-retries(3),                  # Per-backend same-model retry
                                      # budget for transient errors
                                      # (see "Fallback policy" below)
    :timeout(120e0),                  # Seconds per HTTP round-trip
    :parser(-> $text { ... }),        # Optional parser; die to flag
                                      # malformed output (advances
                                      # to the next backend)
    :on-call-complete(-> %p { ... }), # Optional per-call telemetry hook
);

my $result = $task.execute;  # blocks until a backend returns a good
                             # response, or dies if every backend
                             # in the chain fails
```

### Fallback policy

Failures classify into three buckets:

  * **abort** — HTTP 400 / 401 / 402 / 403 / 404.

    Config or account errors (bad API key, payment required, model not found for your region, etc.). Iterating the chain would just produce the same error on every backend, so the Task re-raises immediately with context about which backend failed.

  * **retry-same** — connection drops, HTTP 5xx, or unclassifiable errors.

    Likely transient — a specific OpenRouter upstream provider failed; a retry often routes to a different one. The current backend gets up to `$.max-retries` total attempts (initial + `max-retries - 1` retries) with exponential backoff and jitter capped at 30 s. After the budget is spent the Task advances to the next backend.

  * **advance** — timeout, 429, empty body, parser failure, streamed finish-reason quits (finish_reason 'length' / 'content_filter'), other 4xx.

    Model-specific pathology — a reasoning loop, sanitising rewrite, malformed JSON, rate-limit on this model specifically. The current model's retry budget is short-circuited and the Task moves straight to the next backend. If the chain is exhausted, dies with an "all backend(s) exhausted" summary that includes per-backend error info.

### Truncated output (finish_reason 'length')

Only a *streamed* `'length'` is a transport-level failure. The blocking path the Task uses reports budget exhaustion as an HTTP 200 **success** carrying a partial body, so a truncated payload would sail into `&.parser`, fail there, and (with `:parse-retries` set) be re-rolled as the same over-budget request until the budget was gone.

`:truncation-policy` intercepts it:

  * `'fail'` — record the truncation and advance to the next backend at once, without spending a parse re-roll. `max_tokens` lives in each backend's `Settings`, so only a different backend can change it.

  * `'accept'` — historical behaviour: the partial text is the result.

  * `''` (default) — derive it: `'fail'` when a parser is installed, `'accept'` when not. So `JSONTask` fails fast while prose Tasks are unchanged.

A chain that truncates throws `X::LLM::Data::Inference::Truncated`, a **subclass** of `X::LLM::Data::Inference::Exhausted` — existing handlers and `:on-exhausted` DLQ sinks keep working untouched — whose summary names the levers (raise `max_tokens`, add a larger-budget fallback backend, or shrink the request).

```raku
my $task = LLM::Data::Inference::JSONTask.new(
    :backends($small-budget, $big-budget),
    :$user-prompt,
    :parse-retries(2),   # never burned on a truncation
);
```

### Parser failures advance (behavioural change from pre-fallback)

When `:&parser` is set, a thrown exception from inside the parser is classified as an advance-class failure. The Task does NOT retry the same backend on parser failure — in practice, a model that emits malformed JSON once rarely recovers on a second attempt against the same model, and a chain of `[primary, fallback]` produces cleaner recovery with lower latency.

Consumers on a single-backend Task that previously relied on parser recovery via retry should either (a) add a fallback model to the chain, (b) pass `:backends($primary, $primary)` to preserve the old "try the same model twice" shape on advance-class errors, or (c) build a retry loop at the application layer.

### Retry feedback (`:retry-feedback`)

Callers who do opt into same-backend re-rolls with `:parse-retries(N)` get **blind** re-rolls by default: identical prompt, identical model, no word about what was wrong with the answer just rejected. That is survivable for malformed JSON on a stochastic sampler and hopeless for a *semantic* rejection — a validator that reports "quote 2 does not appear verbatim in the passage" reports exactly that on every re-roll, so the item spends its whole budget failing identically.

`:retry-feedback` (default `False`) makes the re-roll informed: an attempt that follows a parser or validator failure on the **same** backend carries one extra `user` turn,

```text
Your previous response was rejected: <the parser's message>. Respond
again with the same JSON contract, corrected.
```

```raku
my $task = LLM::Data::Inference::JSONTask.new(
    :backends($primary, $fallback),
    :user-prompt('Extract the beat quotes verbatim.'),
    :validator(&quotes-appear-verbatim),
    :parse-retries(2),
    :retry-feedback,
);
```

The rules, each of them a decision:

  * **Replace, never accumulate.** Every attempt re-derives its message list from the pristine prompt, so re-roll 2 argues about re-roll 2's rejection and not re-roll 1's. A growing transcript of complaints would spend more and more prompt on staler and staler objections, and models fixate on the first item in such a list.

  * **Parse and validation failures only.** Truncation is a budget fact rather than a correctable mistake; a network retry-same produced no output to critique; and the pending reason is held **per backend**, so a backend advance starts the next model on the pristine prompt. A model is only ever told about output it actually produced.

  * **No echo of the rejected text.** It is already the model's own last turn in context; the validator's diagnostic is the part the model cannot infer. Every attempt record still carries the raw text for the caller (see `X::LLM::Data::Inference::Exhausted`).

The canned wording names a **JSON contract** — correct for `JSONTask` and hand-rolled JSON parsers, wrong for prose Tasks, which is exactly why the default is off. Telemetry is unchanged: the extra turn is visible only as a larger `prompt-tokens` figure (roughly a hundred tokens) on the retry row.

### Telemetry

`:&on-call-complete` fires once per HTTP round-trip with a hash:

```raku
%(
    attempt       => 1,         # monotonic across the execute call
    backend-index => 0,         # 0-based position within @.backends
    model-name    => 'z-ai/glm-5.1',
    latency-ms    => 1234,
    success       => True,
    stage         => 'network',
    error         => Str,       # present on failure
    error-class   => Str,       # 'http' / 'timeout' / 'connection' /
                                # 'response' / 'unknown' (on failure)
    error-status  => Int,       # HTTP code (when error-class eq 'http')
    # OAI-spec usage — presence-gated, lifted off any
    # LLM::Chat::Backend::Response when the provider supplies them:
    prompt-tokens, completion-tokens, total-tokens,
    model-used, finish-reason,
    # Provider-specific extras — presence-gated, lifted off Response
    # subclasses that expose them (currently
    # LLM::Chat::Backend::Response::OpenRouter):
    cost, generation-id, provider-name, is-byok,
)
```

### classify-error — inspect the policy

The public `classify-error` method maps an error shape to the bucket name for consumers that want to implement the same policy outside the Task:

```raku
my $bucket = $task.classify-error(
    error-class  => 'http',
    error-status => 401,
);
# returns 'abort'

$task.classify-error(error-class => 'timeout');       # 'advance'
$task.classify-error(error-class => 'connection');    # 'retry-same'
$task.classify-error(:parser-failed);                 # 'advance'
```

LLM::Data::Inference::JSONTask
------------------------------

JSON extraction from LLM responses with key validation and optional custom validator. Handles LLMs that wrap JSON in prose by extracting the outermost `{ }` or `[ ]`. Accepts `:$backend` or `:@backends` and threads either through to the inner `Task` unchanged — all fallback semantics come from the Task layer.

```raku
my $task = LLM::Data::Inference::JSONTask.new(
    :backends($primary, $fallback),
    :user-prompt('Give me a character card as JSON.'),
    :required-keys('name', 'description'),    # Missing key → advance
    :validator(-> %h { %h<name>.chars > 0 }), # False return → advance
    :max-retries(3),                          # Same-model retry budget
                                              # for transient errors
);

my %character = $task.execute;
```

LLM::Data::Inference::Router
----------------------------

Query-based routing using Roaring::Tags. Each route is a tag query string paired with a backend. Routes are evaluated in order — first match wins. Orthogonal to the Task-level fallback chain: a Router selects which backend (or chain) to hand to the Task, and the Task handles retries / fallbacks on top.

```raku
my $router = LLM::Data::Inference::Router.new(
    :default-backend($cloud-api),
);

$router.add-route('confidential', $local-model);
$router.add-route('confidential, sensitive', $air-gapped-model);
$router.add-route('genre:technical', $reasoning-model);

my $backend = $router.select-backend($tags, $doc-id);
```

LLM::Data::Inference::PromptBuilder
-----------------------------------

Mustache-style template rendering with `{{variable}}` substitution.

```raku
my $pb = LLM::Data::Inference::PromptBuilder.new(
    :template('Write a {{length}} word {{genre}} story about {{topic}}.')
);
my $prompt = $pb.render(%(:length('500'), :genre('sci-fi'), :topic('AI')));
```

Dies if a `{{variable}}` has no matching key in the vars hash.

AUTHOR
======

Matt Doughty <matt@apogee.guru>

COPYRIGHT AND LICENSE
=====================

Copyright 2026 Matt Doughty

This library is free software; you can redistribute it and/or modify it under the Artistic License 2.0.

