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

  * **advance** — timeout, 429, empty body, parser failure, content-filter quits (finish_reason 'length' / 'content_filter'), other 4xx.

    Model-specific pathology — a reasoning loop, sanitising rewrite, malformed JSON, rate-limit on this model specifically. The current model's retry budget is short-circuited and the Task moves straight to the next backend. If the chain is exhausted, dies with an "all backend(s) exhausted" summary that includes per-backend error info.

### Parser failures advance (behavioural change from pre-fallback)

When `:&parser` is set, a thrown exception from inside the parser is classified as an advance-class failure. The Task does NOT retry the same backend on parser failure — in practice, a model that emits malformed JSON once rarely recovers on a second attempt against the same model, and a chain of `[primary, fallback]` produces cleaner recovery with lower latency.

Consumers on a single-backend Task that previously relied on parser recovery via retry should either (a) add a fallback model to the chain, (b) pass `:backends($primary, $primary)` to preserve the old "try the same model twice" shape on advance-class errors, or (c) build a retry loop at the application layer.

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
    # Provider-reported usage — presence-gated:
    prompt-tokens, completion-tokens, total-tokens,
    cost, model-used, provider-id, finish-reason,
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

