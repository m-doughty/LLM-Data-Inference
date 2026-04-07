[![Actions Status](https://github.com/m-doughty/LLM-Data-Inference/actions/workflows/test.yml/badge.svg)](https://github.com/m-doughty/LLM-Data-Inference/actions)

NAME
====

LLM::Data::Inference - Structured LLM task layer with retry, JSON parsing, and query-based routing

SYNOPSIS
========

```raku
use LLM::Data::Inference;

# Simple blocking LLM call
my $task = LLM::Data::Inference::Task.new(
    :backend($my-backend),
    :system-prompt('You are a helpful assistant.'),
    :user-prompt('What is 2+2?'),
);
say $task.execute;  # "4"

# JSON output with retry
my $json-task = LLM::Data::Inference::JSONTask.new(
    :backend($my-backend),
    :user-prompt('Return a JSON object with name and age.'),
    :required-keys('name', 'age'),
    :max-retries(3),
);
my %result = $json-task.execute;
say %result<name>;  # "Alice"

# Template-based prompts
my $pb = LLM::Data::Inference::PromptBuilder.new(
    :template('Write a {{genre}} story about {{topic}}.')
);
say $pb.render(%(:genre('fantasy'), :topic('dragons')));

# Query-based routing
my $router = LLM::Data::Inference::Router.new(:default-backend($claude));
$router.add-route('nsfw OR violent OR gore', $local-uncensored);
$router.add-route('genre:technical', $reasoning-model);

my $backend = $router.select-backend($tags, $doc-id);
```

DESCRIPTION
===========

LLM::Data::Inference provides a structured task layer on top of LLM::Chat for use in data generation pipelines. It wraps the async LLM::Chat API into blocking calls with automatic retry, JSON extraction, and content-based model routing.

LLM::Data::Inference::Task
--------------------------

Single blocking LLM call with configurable parser and retry.

```raku
my $task = LLM::Data::Inference::Task.new(
    :backend($backend),              # LLM::Chat::Backend (required)
    :system-prompt('Be helpful.'),   # Optional system prompt
    :user-prompt('Hello'),           # User message (required)
    :max-retries(3),                 # Retry on parser failure (default: 3)
    :timeout(120e0),                 # Seconds to wait (default: 120)
    :parser(-> $text { ... }),       # Optional: parse response, die to trigger retry
);

my $result = $task.execute;          # Blocks until response, returns parsed result
```

LLM::Data::Inference::JSONTask
------------------------------

JSON extraction from LLM responses with key validation and optional custom validator. Handles LLMs that wrap JSON in prose by extracting the outermost `{ }` or `[ ]`.

```raku
my $task = LLM::Data::Inference::JSONTask.new(
    :backend($backend),
    :user-prompt('Give me a character card as JSON.'),
    :required-keys('name', 'description'),  # Retry if keys missing
    :validator(-> %h { %h<name>.chars > 0 }),  # Optional extra validation
    :max-retries(3),
);

my %character = $task.execute;
```

LLM::Data::Inference::Router
----------------------------

Query-based routing using Roaring::Tags. Each route is a tag query string paired with a backend. Routes are evaluated in order — first match wins.

```raku
my $router = LLM::Data::Inference::Router.new(
    :default-backend($safe-model),
);

$router.add-route('nsfw', $uncensored-local);
$router.add-route('nsfw, violent', $specialized-model);
$router.add-route('genre:technical', $reasoning-model);
$router.add-route('nsfw OR violent OR gore', $unrestricted);

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

