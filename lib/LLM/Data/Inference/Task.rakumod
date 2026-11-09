=begin pod

=head1 NAME

LLM::Data::Inference::Task - Chat-completion with model-chain fallback

=head1 SYNOPSIS

=begin code :lang<raku>

use LLM::Data::Inference::Task;
use LLM::Chat::Backend;  # or a concrete Backend subclass

# Single backend (legacy shape — unchanged)
my $task = LLM::Data::Inference::Task.new(
    backend     => $backend,
    user-prompt => 'Tell me a joke',
);
my $text = $task.execute;

# Model-chain fallback — try backends in order, falling through on
# model-specific failures (timeout / 4xx-non-auth / empty body /
# parse failure), retry the current backend once on transient errors
# (connection drop / 5xx), abort immediately on config/account errors
# (400 / 401 / 402 / 403 / 404).
my $task2 = LLM::Data::Inference::Task.new(
    backends => [$primary, $secondary, $fallback],
    user-prompt => 'Write a scene',
);
my $text2 = $task2.execute;

=end code

=head1 DESCRIPTION

One-shot LLM invocation with a robust retry + fallback policy.
Callers supply either a single C<:$backend> (legacy, backward-compat)
or an ordered C<:@backends> chain; the Task iterates the chain with
a three-bucket error classifier:

=head2 Error buckets

=item B<abort> — HTTP 400/401/402/403/404. These are config, account, or
access errors; iterating the chain is wasted effort because the same
error repeats. Re-raises immediately with context.

=item B<retry-same> — connection drop, 5xx, or an unclassifiable error.
Likely transient (a specific OpenRouter upstream provider failed; a
retry often routes to a different one). The current backend gets one
additional attempt, then the Task advances to the next backend if
it still fails.

=item B<advance> — timeout, 429 rate-limit, empty-body response,
finish-reason quit (length / content_filter), or parser validation
failure. These are model-specific: the current model ran into a
pathology (reasoning loop, sanitisation, malformed JSON). Skip to
the next backend.

The "finish-reason quit" case above is B<streaming-only>. A streamed
C<'length'> / C<'content_filter'> makes the backend quit the supply
with error class C<'response'>, which lands in this bucket. The
B<blocking> C<chat-completion> path the Task actually uses does not
fail on either: a completion cut off by C<max_tokens> comes back as an
ordinary HTTP 200 success carrying a partial body, with the reason
only visible as C<$response.finish-reason>. See L</Truncated output>
for how that is intercepted.

=head2 Per-backend attempt budget

Each backend gets up to C<$.max-retries> HTTP round-trips (default 3):
one initial attempt plus C<max-retries - 1> retries reserved for
retry-same-class errors. An advance-class error on any attempt
short-circuits the budget and moves straight to the next backend.
The chain is exhausted when every backend has been tried.

Retries use exponential backoff with jitter (C<2^(retry-1) + [0,0.5)>
seconds, capped at 30 s) to avoid thundering-herd on concurrent
workers.

=head2 Empty-completion recovery

C<:empty-retries(N)> permits at most N additional calls across one
C<execute>, independently of backend, network and parser budgets. Default
0 preserves existing behavior, including whitespace-only prose results.
Opted-in tasks treat whitespace-only answers as empty. Only successful
responses with no reasoning or tool calls and a C<stop> or absent finish
reason qualify. Each retry preserves the previous request unchanged.
Cancellation is checked before retrying and while the call is in flight.

Empty failures retain content-free response metadata in attempt records
and summaries. Missing metadata is absent in records and shown as
C<unknown> in summaries; no cause is inferred from those values.

An empty body whose Response carries thinking is never retried this
way; L</Empty responses that carry the answer in the thinking> is the
rule for that shape.

=head2 Empty responses that carry the answer in the thinking

Some servers file a model's whole output as thinking. llama.cpp's
chat parser does it when the chat template left the think block open
and the model never emitted the closing tag: the answer sits in
C<reasoning_content>, the content is empty, and the finish reason is
an ordinary C<stop>. DeepSeek's reference parser rejects that
completion as malformed, so nothing upstream repairs it, and to this
Task it looks exactly like a model that thought until it ran out of
things to say — except that the thinking is a complete answer.

C<:empty-response-uses-reasoning> (default False) takes the thinking
as the response in exactly that case: the call succeeded, the body is
empty by the same rule L</Empty-completion recovery> uses, the finish
reason is C<stop> or absent, the Response carries no tool calls, and
the thinking is non-empty. The thinking then goes through the parser
as the body would have, or is returned as the text when there is no
parser. A C<length> finish is never taken: a think block the budget
cut off is not an answer, and the truncation rule has already dealt
with it.

A salvaged answer the parser rejects is a parser failure, not an
empty response: C<:parse-retries> and C<:retry-feedback> apply, the
attempt record carries C<answer-source =E<gt> 'reasoning'> and the
thinking under C<reasoning-text>, and C<raw-text> is left out rather
than repeating the same text. The error line says the parser ran on
the thinking. C<:empty-retries> never fires once the thinking is
taken, which is no change: it has always required an empty reasoning
trace.

A salvage that parses leaves no attempt record, since records are
failure-only; the C<&on-call-complete> payload says it happened (see
L</Telemetry>).

=begin code :lang<raku>

my %answer = LLM::Data::Inference::JSONTask.new(
    backend       => $llama-cpp,
    system-prompt => 'Reply with a JSON object: {"prompt": ...}',
    user-prompt   => $scene,
    required-keys => <prompt>,
    :empty-response-uses-reasoning,
    on-call-complete => -> %row {
        note 'answer taken from the thinking'
            if %row<used-reasoning-as-response>;
    },
).execute;

=end code

=head2 Parser failures

When C<:&parser> is set, the parser is invoked on the raw text. A thrown
exception inside C<&parser> is treated as an advance-class failure
(different model may produce parseable output). By default the Task
does NOT retry the same backend on parser failure — the previous
per-backend retry loop produced identical malformed output enough times
in practice (batch pipelines, deterministic-ish sampling) that burning
budget on same-model parse retries was net-negative.

Interactive callers on stochastic samplers can opt into same-backend
re-rolls with C<:parse-retries(N)>: each parser failure consumes one
re-roll (a fresh completion from the same backend, no backoff) before
the advance rule kicks in. The raw model output of every failed parse
is preserved on the thrown C<X::LLM::Data::Inference::Exhausted>
(see L<LLM::Data::Inference::Exceptions>), so callers can surface
I<what the model actually said> instead of only the parse error.

=head2 Thinking on attempt records

A reasoning model's thinking arrives on the Response's
C<reasoning-text>, never in its content, and the Task never sends it
back. It does keep it: every attempt record whose Response carried
some thinking gets a C<reasoning-text> key holding it verbatim, on
the same presence rule as C<raw-text> (the key is absent, not
undefined, when there was none). Parser failures, truncations, empty
bodies and network failures that still produced a Response all
qualify.

The empty body is the case that earns this. A thinking model that
spends its whole completion inside its reasoning returns a C<200>
with nothing in the content, so there is no C<raw-text> to show, and
the summary line can only count characters (see
L</Empty-completion recovery>). The trace itself is the one piece of
evidence about what the model was doing, and it lives on the record
for a failure UI to show. Summaries and telemetry stay content-free:
the trace is on the exception, not in any message.

One record shape goes further: a parser failure whose input was the
thinking rather than the body (see
L</Empty responses that carry the answer in the thinking>) carries
C<answer-source =E<gt> 'reasoning'> and no C<raw-text>, so a viewer
shows the thinking once, under the right heading. Every other record
has no C<answer-source> key at all.

=begin code :lang<raku>

CATCH {
    when X::LLM::Data::Inference::Exhausted {
        for .attempts -> %a {
            note "model %a<model> thought:\n%a<reasoning-text>"
                if %a<reasoning-text>:exists;
        }
    }
}

=end code

=head2 Retry feedback

A parse re-roll is B<blind> by default: the same prompt goes back to
the same model, and nothing tells it what was wrong with the answer it
just gave. That is fine for malformed JSON on a stochastic sampler — a
fresh roll usually closes its brackets — but useless for a I<semantic>
rejection. A validator that says "quote 2 does not appear verbatim in
the passage" will keep saying it, three times, and the Task will
dead-letter an item the model could have fixed on the second try.

C<:retry-feedback> (opt-in, default C<False>) makes the re-roll
informed: after a parser or validator failure, the next attempt against
B<the same backend> carries one extra C<user> turn naming the rejection.

=begin code :lang<raku>

my $task = LLM::Data::Inference::JSONTask.new(
    :backends($primary, $fallback),
    :$user-prompt,
    :validator(&quotes-appear-verbatim),
    :parse-retries(2),
    :retry-feedback,       # re-rolls now say WHY the last one lost
);

=end code

The appended turn reads:

=begin code :lang<text>

Your previous response was rejected: <the parser's message>. Respond
again with the same JSON contract, corrected.

=end code

Three properties are worth being explicit about, because each one is a
decision rather than an accident:

=item B<Replace, never accumulate.> Each attempt derives its message
list from the pristine C<@messages>, so re-roll 2 carries re-roll 2's
rejection and B<not> re-roll 1's. The alternative — appending to a
growing transcript — would spend an ever-larger prompt on
progressively staler complaints, and models reliably fixate on the
first item in such a list.

=item B<Parse failures only.> The feedback slot is filled in the
parser-failure branch and nowhere else. A truncation is a budget fact,
not a mistake the model can correct by trying harder (see
L</Truncated output>); a network retry-same never produced output to
critique; and the slot lives B<per backend>, so a backend advance
starts the next model on the pristine prompt — its predecessor's
mistakes are not evidence about anything it did. (The slot is sticky
until the backend changes: if a parse failure is followed by a
transient network error, the retry after that error still carries the
pending feedback, because the rejection it describes is still the last
thing the model actually said.)

=item B<No echo of the rejected output.> The feedback names the error,
not the text that produced it. The model has its own last turn in
context on every provider worth using, so echoing it back doubles the
prompt to restate what the model already knows; and validator
diagnostics in this ecosystem are written to identify the failing item
precisely ("beat 3, quote 2: ..."), which is the part the model cannot
infer. If a caller wants the raw text in the loop, it is on every
attempt record already — write a parser that folds it into its own
message.

The canned wording names a B<JSON contract>, which makes it correct for
C<JSONTask> and for hand-rolled JSON parsers, and wrong for prose
Tasks. That is deliberate: prose Tasks simply do not opt in. (Callers
who want feedback with different wording can raise it from inside the
parser instead, since the parser's message is what gets quoted.)

Telemetry is unchanged — C<&on-call-complete> carries no message-count
field, so the extra turn shows up only as a larger C<prompt-tokens>
figure on the retry row. Budget roughly a hundred extra prompt tokens
per informed re-roll.

=head2 Truncated output

A blocking completion that runs out of completion budget is not an
error at the transport level: HTTP 200, a well-formed body, a partial
answer, and C<finish_reason> C<'length'>. Left alone, the Task would
hand that half-finished text to C<&.parser>, watch it fail, and — with
C<:parse-retries> set — re-roll the I<identical doomed request> until
the budget was spent, because a request that overran once will overrun
again at the same C<max_tokens>.

C<$.truncation-policy> decides what happens instead:

=item C<'fail'> — a C<'length'> finish reason is recorded as a failure
and the Task advances to the next backend B<immediately>, without
consuming a parse re-roll. C<max_tokens> is per-backend
C<Settings> state, so the next backend in the chain can legitimately
have a bigger budget and succeed where this one could not — but a
same-backend re-roll cannot. If the whole chain truncates, the Task
throws C<X::LLM::Data::Inference::Truncated> (a subclass of
C<Exhausted>, so existing handlers are unaffected) and the summary
names the levers.

=item C<'accept'> — historical behaviour. The partial text is the
result: returned as-is when there is no parser, handed to the parser
when there is.

=item C<''> (the default) — B<derive> the policy: C<'fail'> when
C<&.parser> is set, C<'accept'> when it is not. A truncated payload
headed for a parser can never parse, so failing fast only removes a
guaranteed-waste path; truncated prose, on the other hand, is
frequently still the product (continuations, style passes), so
parser-less Tasks keep their historical behaviour bit-for-bit. Set the
attribute explicitly to override in either direction.

C<LLM::Data::Inference::JSONTask> always installs a parser, so it
defaults to C<'fail'>.

The empty-body rule is checked B<after> truncation, deliberately: a
reasoning model that spends its entire budget inside
C<< <think>…</think> >> returns 200 with an empty body and
C<finish_reason> C<'length'>. That is budget exhaustion, and reporting
it as "empty response body" sends the reader hunting the wrong bug.

=begin code :lang<raku>

# Prose task that wants the partial text anyway (the default already
# does this, since no parser is set — spelled out for clarity).
my $prose = LLM::Data::Inference::Task.new(
    :$backend, :$user-prompt, truncation-policy => 'accept',
);

# JSON task on a chain whose second backend has a bigger budget.
my $json = LLM::Data::Inference::JSONTask.new(
    backends => [$small-budget, $big-budget], :$user-prompt,
    parse-retries => 2,   # never burned on a truncation
);

=end code

=head2 Typed exhaustion and levers

When the chain runs out, the Task throws one of three types — all of
them C<X::LLM::Data::Inference::Exhausted> or a subclass of it, so a
handler that never heard of the subclasses is unaffected:

=item C<Truncated> — at least one attempt was cut off by C<max_tokens>
(only reachable under C<truncation-policy> C<'fail'>; see above).

=item C<TimedOut> — at least one attempt died on a response deadline:
C<$.timeout> firing while the Task polled the in-flight response, or a
backend reporting error class C<'timeout'> itself.

=item C<Exhausted> — everything else.

B<Precedence: C<Truncated> wins over C<TimedOut>> when a chain saw
both. It is the more specific diagnosis, and it is the deterministic
one — it declares C<item-retryable> C<False> (see
L<LLM::Data::Inference::Exceptions>), which tells an orchestrator to
stop re-running the item rather than buying back retries a truncation
guarantees will fail. C<TimedOut> makes no such claim: a deadline miss
is transient, so it keeps its full item budget.

The C<summary> (and therefore C<.message>) gains one B<lever line> per
pathology observed, appended after the per-attempt error list. A chain
that both truncated and timed out gets both lines — they name
different fixes:

=begin code :lang<text>

LLM::Data::Inference::Task: all 2 backend(s) exhausted.
  [backend 0 small] truncated: output was cut off by the max_tokens …
  [backend 1 slow] timeout: response timed out after 300s
  At least one attempt was cut off by the completion budget: raise …
  At least one attempt hit the 300-second response deadline before …

=end code

Summaries of chains that hit neither are byte-identical to what they
have always been.

=head2 Telemetry

Every HTTP round-trip fires C<&on-call-complete> (if provided), with
the attempt number, model name, backend index (position in the chain),
latency, success flag, usage data, and error details. Parser failures
do not fire separately — they're classified from the preceding
telemetry row. Every payload also carries
C<used-reasoning-as-response>, a Bool that is True when the row's
empty body is being answered from its thinking (see
L</Empty responses that carry the answer in the thinking>) — the only
trace a salvage that parses leaves behind.

C<&on-exhausted> (if provided) fires exactly once, on the execute
thread, immediately before the chain throws
C<X::LLM::Data::Inference::Exhausted> B<or either of its subclasses>
(see L</Typed exhaustion and levers>) — never on success, and never
on the C<Cancelled> path. The payload is C<stage =E<gt> 'exhausted'>,
C<attempts> (the same C<@attempt-records> array the thrown exception
carries — including C<raw-text> for parse failures and
C<reasoning-text> wherever the model thought out loud), C<summary> (the
same aggregate string as the exception's C<.message>), and
C<backends> (the chain length). Like C<&on-call-complete>, a hook
that throws is shielded — it can never suppress or alter the
C<Exhausted> throw that always follows.

=begin code :lang<raku>

my $task = LLM::Data::Inference::Task.new(
    backends      => [$primary, $fallback],
    user-prompt   => 'q',
    on-exhausted  => -> %info {
        $dlq.append: %(
            attempts => %info<attempts>,
            summary  => %info<summary>,
        );
    },
);

=end code

=head2 Cooperative cancellation

C<:&is-cancelled> is polled at every decision point: before each
round-trip, inside the in-flight completion poll (the pending response
is aborted via C<Backend.cancel>, exactly like the timeout path),
between backoff sleeps, and before parse re-rolls / backend advances.

Aborting through C<Backend.cancel> rather than C<Response.cancel> is
what makes the cancel reach the server: against KoboldCpp it fires
C<POST /api/extra/abort> alongside the local stream close, and against
an OpenAI-compatible backend it hangs up the HTTP exchange at the
socket instead, which openai-compat servers treat as their own abort
signal — either way, a cancelled Task stops the generation instead of
only looking away from it. A backend with neither upstream signal
degrades to a local close.
When it returns True the Task throws
C<X::LLM::Data::Inference::Cancelled> (see
L<LLM::Data::Inference::Exceptions>) carrying whatever attempt records
already exist. Worst-case latency between the hook flipping and the
throw is one poll interval (~10 ms in-call, ~250 ms mid-backoff) —
a cancelled Task never burns a full retry chain.

=begin code :lang<raku>

my $task = LLM::Data::Inference::Task.new(
    backend      => $backend,
    user-prompt  => 'Reconcile this exchange as JSON',
    parser       => &json-parser,
    parse-retries => 2,
    is-cancelled => -> { $job.cancelled },
);

=end code

=head1 BACKWARD COMPATIBILITY

Callers passing C<:$backend> only see no behaviour change beyond the
new retry policy. The C<$.backend> accessor remains available and
always points at the first backend in the chain; legacy consumers that
read it (e.g. to label log messages with the model) keep working.

=end pod

use LLM::Chat::Backend;
use LLM::Chat::Backend::Response;
use LLM::Chat::Retry;
use LLM::Chat::ExceptionText;
use LLM::Data::Inference::Exceptions;
use LLM::Chat::Conversation::Message;

unit class LLM::Data::Inference::Task;

has LLM::Chat::Backend $.backend;
has LLM::Chat::Backend @.backends;
has Str $.system-prompt;
has Str:D $.user-prompt is required;
#|( Per-backend same-model attempt budget when the failure is a
    retry-same-class error (connection drop, 5xx). Includes the
    initial attempt — C<max-retries = 3> means one initial + up to
    two retries on that backend before the Task advances. Parser
    failures and model-specific errors (timeout, 429, 4xx, empty
    body, finish-reason quit) ignore this budget and advance
    immediately. Default matches the pre-fallback behaviour so
    legacy single-backend callers see no change on transient
    network failures. )
has Int:D $.max-retries = 3;
has &.parser;

#|( Same-backend re-roll budget for B<parser> failures, separate from
    C<max-retries> (which covers transient network errors). Default 0
    preserves the historical behaviour: a parse failure advances to
    the next backend immediately. Set it above 0 for interactive
    callers on stochastic samplers, where a fresh sample from the
    same model frequently parses fine — batch pipelines with
    deterministic-ish sampling should leave it at 0 (identical
    malformed output re-rolled is wasted spend; see the module Pod). )
has UInt:D $.parse-retries = 0;

#| Additional eligible empty-completion calls across the whole execute.
has UInt:D $.empty-retries = 0;

#|( Take the thinking as the response when the body is empty. Gated on
    a successful call with a C<stop> or absent finish reason, no tool
    calls and a non-empty C<reasoning-text>; the thinking then meets
    the parser exactly as the body would have, and a rejection is a
    parser failure (re-rolls, feedback) rather than an empty one.
    Default False: a reasoning model that genuinely thought until it
    ran out of budget must keep failing as "empty response body". See
    the "Empty responses that carry the answer in the thinking"
    section of the module Pod. )
has Bool:D $.empty-response-uses-reasoning = False;

#|( Tell the model B<why> a parse re-roll is happening. When True, an
    attempt that follows a parser/validator failure on the SAME backend
    carries one extra C<user> turn quoting the rejection reason, which
    turns a blind re-roll into a correction pass (the difference between
    "3 identical failures" and "fixed on attempt 2" for semantic
    validators). Only meaningful together with C<$.parse-retries> — with
    a zero re-roll budget there is no second attempt to inform.

    Each re-roll REPLACES the previous feedback rather than accumulating
    a transcript of complaints, and the slot is per-backend, so
    truncation advances, network retry-sames and backend advances never
    inherit it. Default False: the canned wording names a JSON contract,
    so prose Tasks must not get it by accident. See the "Retry feedback"
    section of the module Pod. )
has Bool:D $.retry-feedback = False;

#|( What to do when a completion comes back cut off by the token
    budget (HTTP 200, partial body, C<finish_reason> 'length'):

      * 'fail'   — record the truncation and advance to the next
                   backend at once, without consuming a parse
                   re-roll; throw C<Truncated> if the chain exhausts.
      * 'accept' — historical behaviour: treat the partial text as
                   the result (parse it, or return it raw).
      * ''       — derive it (the default): 'fail' when C<&.parser>
                   is set, 'accept' when it is not.

    The derived default is the whole point of the empty-string
    sentinel: truncated JSON can never parse, so re-rolling the same
    over-budget request is guaranteed waste, while truncated prose is
    often still usable and must keep behaving exactly as it always
    has. See the "Truncated output" section of the module Pod. )
has Str:D $.truncation-policy where { $_ eq any('', 'fail', 'accept') } = '';

#|( Caller-side response deadline, in seconds, applied to EVERY
    round-trip (not to the execute as a whole). When it fires the
    pending response is aborted through C<Backend.cancel> and the
    attempt is classified C<'timeout'> → advance to the next backend.
    A chain exhausted with at least one such attempt throws
    C<X::LLM::Data::Inference::TimedOut> and names this value in the
    summary's lever line. )
has Num:D $.timeout = 120e0;

#|( Optional telemetry hook. Fires once per HTTP round-trip,
    including failures. Payload keys:
      * attempt        — 1-based, monotonic across the whole execute
      * backend-index  — 0-based position of the backend within @.backends
      * model-name     — try { $backend.model } // 'unknown'
      * latency-ms     — end-to-end wall time
      * success        — Bool
      * error          — Str, present when success is False
      * error-class    — Str, Task-side classification (see module Pod)
      * error-status   — Int, HTTP status when error-class eq 'http'
      * stage          — 'network'
      * prompt-tokens / completion-tokens / total-tokens /
        model-used / finish-reason — presence-gated, lifted off
        any C<LLM::Chat::Backend::Response> when the provider
        supplies them.
      * answer-chars / answer-visible-chars / reasoning-chars — content-free counts,
        present when the Response exposes the corresponding text.
      * cost / generation-id / provider-name / is-byok —
        presence-gated, lifted off Response subclasses that expose
        them (currently C<LLM::Chat::Backend::Response::OpenRouter>).
        On other backends these keys are simply absent from the
        payload. )
has &.on-call-complete;

#|( Optional exhaustion hook. Fires exactly once, on the execute
    thread, immediately before C<X::LLM::Data::Inference::Exhausted>
    is thrown — never on success, and never on the C<Cancelled> path.
    Payload keys:
      * stage     — 'exhausted'
      * attempts  — the same C<@attempt-records> the thrown exception
                    carries (identical array, not a copy)
      * summary   — the same aggregate summary string as the
                    exception's C<.message>
      * backends  — count of backends in the chain that was tried
    Shielded exactly like C<&on-call-complete>: a hook that throws
    is caught and noted, never suppressing or altering the
    C<Exhausted> throw that follows. )
has &.on-exhausted;

#|( Optional cancellation hook, polled at every decision point (loop
    top, in-flight response poll, mid-backoff). Returning True makes
    the Task abort any pending response and throw
    C<X::LLM::Data::Inference::Cancelled>. Must be cheap and
    thread-safe — it runs on the Task's worker at ~10 ms cadence
    while a call is in flight. )
has &.is-cancelled;

submethod TWEAK() {
	# Accept :$backend, :@backends, or both. Normalise so @!backends
	# is always populated and $!backend points at its head.
	if @!backends.elems == 0 {
		die "LLM::Data::Inference::Task requires :backend or :backends"
			unless $!backend.defined;
		@!backends = ($!backend,);
	}
	$!backend = @!backends[0] without $!backend;
}

#|( Classify a failure by its structured error shape (as set on
    C<Response._set-error-info> by the backend) into one of three
    buckets: C<'abort'>, C<'retry-same'>, or C<'advance'>. See the
    module Pod for the full rule table. Pure function — exposed on
    the class for test ergonomics, but the retry loop is the only
    caller in normal use.

    Since 0.9.0 the rule table itself lives in
    L<LLM::Chat::Retry>, so a second executor can share the policy
    rather than copy it; this method is the (unchanged) public face
    of it and delegates verbatim. Deliberately kept rather than
    deprecated: it is part of this class's tested surface, and a
    caller holding a Task should not have to reach for another
    distribution to ask "what would you do with this error?". )
method classify-error(
	Str :$error-class,
	Int :$error-status,
	Bool :$parser-failed = False,
	--> Str
) {
	LLM::Chat::Retry::classify-error(
		:$error-class, :$error-status, :$parser-failed,
	);
}

#|( The one extra C<user> turn a C<:retry-feedback> re-roll carries.

    Names the rejection and re-states the contract, and deliberately
    does NOT echo the rejected output: the model's own last turn is
    already in context, so quoting it back doubles the prompt to tell
    the model something it knows, while the parser's message is the
    part it cannot infer (this ecosystem's validators name the failing
    item precisely — "beat 3, quote 2: ...").

    "the same JSON contract" is why C<$.retry-feedback> defaults to
    False: it is right for JSON parsers and wrong for prose, and prose
    Tasks therefore never opt in. Pure function, no state — the
    per-backend feedback slot lives in C<execute>. )
my sub feedback-message(Str:D $error --> Str:D) {
	"Your previous response was rejected: $error. "
		~ "Respond again with the same JSON contract, corrected.";
}

method execute(--> Any) {
	my @messages;
	if $!system-prompt.defined {
		@messages.push: LLM::Chat::Conversation::Message.new(
			role => 'system', content => $!system-prompt,
		);
	}
	@messages.push: LLM::Chat::Conversation::Message.new(
			role => 'user', content => $!user-prompt,
	);

	my Int:D $attempt = 0;
	my UInt $empty-retries-left = $!empty-retries;
	my @errors;
	my @attempt-records;
	my Str:D $truncation-policy = self!effective-truncation-policy;
	# Sticky across the whole chain: one truncated attempt anywhere is
	# enough to make the exhaustion Truncated-typed and to earn the
	# summary its "raise the budget" line.
	my Bool $saw-truncation = False;
	# Its deadline twin. Sticky for the same reason, and a flag rather
	# than something derived from @attempt-records at the end because
	# the records keep only the rendered error STRING — error-class is
	# live exactly once, in the network-failure branch below, and
	# re-parsing prose to recover it would be a worse contract than
	# setting a bit while the classification is in hand.
	my Bool $saw-timeout = False;
	sub record-failure(Int $backend-index, Str $model, Str $error,
		Str :$raw-text, Str :$reasoning-text, Str :$answer-source,
		:%diagnostics) {
		@errors.push: $error;
		# Shared record shape (raw-text key ABSENT, not undefined, when
		# there was no body worth keeping) — see LLM::Chat::Retry.
		my %record = attempt-record(
			:$backend-index, :model($model), :$error, :$raw-text,
		);
		# Same rule for the thinking: the key exists only when the model
		# produced some. An empty body with a full reasoning trace is the
		# common shape of a thinking model that never reached its answer,
		# and the trace is the only evidence of what it was doing.
		%record<reasoning-text> = $reasoning-text
			if $reasoning-text.defined && $reasoning-text.chars;
		# Present only when the parser's input was the thinking rather
		# than the body, so ordinary records keep their exact shape.
		%record<answer-source> = $answer-source
			if $answer-source.defined && $answer-source.chars;
		%record<response-metadata> = %diagnostics.Hash if %diagnostics.elems;
		@attempt-records.push: %record;
	}
	# The thinking a Response carries, or an undefined Str when it has
	# none (a non-reasoning model, a call that failed before any arrived,
	# or no Response at all). Never the content.
	sub reasoning-of($response --> Str) {
		return Str without $response;
		my $text = try { $response.reasoning-text };
		($text.defined && $text.chars) ?? $text.Str !! Str;
	}
	sub cancelled-now(--> Bool) {
		&!is-cancelled.defined && ?&!is-cancelled();
	}
	sub throw-cancelled(--> Nil) {
		X::LLM::Data::Inference::Cancelled.new(
			attempts => @attempt-records,
		).throw;
	}

	for @!backends.kv -> $i, $backend {
		my $same-retries-left = $!max-retries > 0 ?? $!max-retries - 1 !! 0;
		my $parse-retries-left = $!parse-retries;
		my $model-name = (try { $backend.model }) // 'unknown';
		# Pending retry feedback for THIS backend: the rejection reason
		# from its last parse failure, or undefined when the next call is
		# a first impression. Declared inside the backend loop so a
		# backend advance resets it for free — the next model must not
		# inherit its predecessor's mistakes.
		my Str $feedback;

		loop {
			# One check point covers the initial call, post-backoff
			# retries, parse re-rolls, and backend advances — every
			# path re-enters the loop top before touching the network.
			throw-cancelled if cancelled-now;
			$attempt++;
			# Derived from the PRISTINE @messages every single iteration:
			# that is precisely what makes each re-roll REPLACE the
			# previous feedback instead of stacking a transcript of
			# rejections. @messages itself is never mutated.
			my @call-messages = ($!retry-feedback && $feedback.defined)
				?? (
					|@messages,
					LLM::Chat::Conversation::Message.new(
						role => 'user', content => feedback-message($feedback),
					),
				)
				!! @messages;
			my %call = self!call-blocking(@call-messages, $backend);
			# The finish reason, '' when the provider reported none (or
			# there is no Response). Read once, here: the truncation rule,
			# the empty-body rule and the thinking salvage all key on it.
			my Str $finish = ((%call<response>.defined
				?? %call<response>.finish-reason
				!! Str) // '');
			# Decided before telemetry fires so the row can say so — a
			# salvage that parses leaves no other trace.
			my Bool $uses-reasoning =
				self!takes-reasoning-as-response(%call, $finish);
			self!fire-telemetry(
				%call,
				:$attempt,
				:backend-index($i),
				:$model-name,
				:used-reasoning-as-response($uses-reasoning),
			);
			# The in-call poll aborted the response — the round-trip
			# is on the telemetry record, but its outcome is the
			# caller's withdrawal, not a chain failure.
			throw-cancelled if %call<cancelled>;

			# Network-layer failure
			unless %call<success> {
				my $bucket = self.classify-error(
					error-class  => %call<error-class>,
					error-status => %call<error-status>,
				);
				# Covers both flavours of "ran out of time": the
				# caller-side deadline in !call-blocking, and a
				# backend that classified its own failure as a
				# timeout. Both are fixed by the same levers.
				$saw-timeout = True
					if (%call<error-class> // '') eq 'timeout';
				record-failure($i, $model-name,
					"[backend $i $model-name] "
					~ "{%call<error-class> // 'unknown'}"
					~ (%call<error-status>.defined ?? " {%call<error-status>}" !! '')
					~ ": {%call<error>}",
					reasoning-text => reasoning-of(%call<response>));

				given $bucket {
					when 'abort' {
						die "LLM::Data::Inference::Task: aborting on "
							~ "{%call<error-class> // 'unknown'}"
							~ (%call<error-status>.defined ?? " {%call<error-status>}" !! '')
							~ " from backend $i ($model-name): {%call<error>}";
					}
					when 'retry-same' {
						if $same-retries-left > 0 {
							# Exponential backoff with jitter so N
							# parallel workers hitting the same upstream
							# don't thundering-herd after a rate-limit
							# burst. Retry number is derived from budget
							# consumed (max-retries minus remaining minus
							# one); capped at 30 s. Slept in chunks so a
							# cancel doesn't wait out the full backoff —
							# sleep-with-cancel returns False the moment
							# the hook flips, and only THIS layer knows
							# which exception that becomes.
							my $retry-n = $!max-retries - $same-retries-left;
							my Num $wait = retry-backoff($retry-n);
							$same-retries-left--;
							throw-cancelled unless sleep-with-cancel(
								$wait, cancelled => &cancelled-now,
							);
							next;
						}
						last;  # budget spent, advance
					}
					default { last }  # 'advance'
				}
			}

			# HTTP succeeded — inspect the body.
			my Str $text = %call<text>;

			# Budget exhaustion, which the blocking path reports as a
			# SUCCESS: 200, a well-formed body, and finish_reason
			# 'length'. Checked before the empty-body rule on purpose —
			# a reasoning model that spends its whole budget thinking
			# returns an empty body with reason 'length', and calling
			# that "empty response body" points the reader at the wrong
			# bug.
			if $finish eq 'length' && $truncation-policy eq 'fail' {
				$saw-truncation = True;
				# Keep whatever partial output arrived so the caller
				# can see how far the model got; an empty body has
				# nothing worth recording.
				my Str $partial = ($text.defined && $text.chars)
					?? $text !! Str;
				record-failure($i, $model-name,
					"[backend $i $model-name] truncated: output was cut "
					~ "off by the max_tokens completion budget "
					~ "(finish_reason 'length')",
					raw-text => $partial,
					reasoning-text => reasoning-of(%call<response>));
				# Advance WITHOUT touching $parse-retries-left: the
				# same backend re-rolling the same over-budget request
				# will overrun again at the same max_tokens, while the
				# next backend may carry a bigger budget.
				last;
			}

			# Where the parser's input came from: the body, or the
			# thinking standing in for an empty body (the attribute's Pod
			# has the rule). Decided before any failure is recorded, so a
			# salvage that parses leaves nothing in @errors or the records.
			my Str $answer-source = 'content';
			if self!empty-body($text) {
				if $uses-reasoning {
					$text = %call<response>.reasoning-text.Str;
					$answer-source = 'reasoning';
				}
				else {
					my %diagnostics = self!response-metadata(%call<response>);
					my Str $details = <answer-chars reasoning-chars finish-reason
						prompt-tokens completion-tokens total-tokens>.map({
						"$_=" ~ (%diagnostics{$_} // 'unknown')
					}).join(', ');
					record-failure($i, $model-name,
						"[backend $i $model-name] empty response body ($details)",
						reasoning-text => reasoning-of(%call<response>),
						:%diagnostics);
					my $response = %call<response>;
					if $empty-retries-left > 0 && $response.defined
						&& ($finish eq '' || $finish eq 'stop')
						&& !($response.reasoning-text // '').chars
						&& !$response.has-tool-calls {
						throw-cancelled if cancelled-now;
						$empty-retries-left--;
						next;
					}
					last;  # advance
				}
			}

			# No parser — the text (body, or the thinking standing in for
			# an empty body) is the result.
			return $text unless &!parser.defined;

			my $parsed;
			my $parse-failed = False;
			my $parse-error;
			try {
				$parsed = &!parser($text);
				CATCH {
					default {
						$parse-failed = True;
						$parse-error  = exception-text($_);
					}
				}
			}
			return $parsed unless $parse-failed;

			# A rejected salvage names its input and leaves raw-text out:
			# the thinking is already on the record as reasoning-text, and
			# repeating it under a second key would show it twice.
			my %source-args = $answer-source eq 'reasoning'
				?? %( :$answer-source )
				!! %( raw-text => $text );
			record-failure($i, $model-name,
				"[backend $i $model-name] parser"
				~ ($answer-source eq 'reasoning'
					?? ' (on the thinking, the response body being empty)'
					!! '')
				~ ": $parse-error",
				|%source-args,
				reasoning-text => reasoning-of(%call<response>));
			# The ONE place feedback is produced. Overwrites whatever was
			# pending, so the next re-roll argues about the newest
			# rejection only. Set unconditionally (not just when a re-roll
			# is left) — a spent budget takes `last` below and the slot
			# dies with the backend either way.
			$feedback = $parse-error if $!retry-feedback;
			if $parse-retries-left > 0 {
				# Stochastic samplers usually produce parseable
				# output on a fresh roll; no backoff — the backend
				# just served us fine.
				$parse-retries-left--;
				next;
			}
			last;  # advance
		}
	}

	# A lever line only appears when its condition was actually
	# recorded, so summaries of ordinary failures are byte-identical
	# to what they always were. A chain that hit both pathologies
	# gets BOTH lines: they name different fixes, and suppressing one
	# because the other applies would hide half the diagnosis.
	my Str:D $lever = ($saw-truncation
		?? "\n  At least one attempt was cut off by the completion "
			~ "budget: raise that backend's max_tokens completion "
			~ "budget, add a fallback backend with a larger one, or "
			~ "shrink the request."
		!! '')
		~ ($saw-timeout
		?? "\n  At least one attempt hit the {$!timeout}-second "
			~ "response deadline before completing: raise the task "
			~ "timeout, use a faster backend, or shrink the response."
		!! '');
	my Str:D $summary =
		"LLM::Data::Inference::Task: all {@!backends.elems} backend(s) exhausted.\n"
		~ @errors.map({ "  $_" }).join("\n")
		~ $lever;
	# Fires identically for every exception type — a DLQ sink sees the
	# same payload shape it always has, with the lever lines included.
	self!fire-exhausted(@attempt-records, $summary);
	# Precedence: Truncated beats TimedOut when the chain saw both.
	# It is the more specific diagnosis AND the deterministic one —
	# it declares `item-retryable` False, so an orchestrator stops
	# re-running the item, which is exactly the right call when the
	# same chain also timed out somewhere. Downgrading it to TimedOut
	# would buy back retries that a truncation guarantees will fail.
	my $ex-type = $saw-truncation
		?? X::LLM::Data::Inference::Truncated
		!! ($saw-timeout
			?? X::LLM::Data::Inference::TimedOut
			!! X::LLM::Data::Inference::Exhausted);
	$ex-type.new(
		attempts => @attempt-records,
		summary  => $summary,
	).throw;
}

#|( Resolve C<$.truncation-policy>'s empty-string sentinel into the
    concrete policy this Task runs under. Explicit 'fail'/'accept'
    pass through; '' derives 'fail' when a parser is installed (a
    truncated payload can never parse, so advancing at once is the
    only non-wasteful move) and 'accept' when one is not (truncated
    prose is frequently still the product, and parser-less callers
    must keep their historical behaviour). Called once per execute. )
method !effective-truncation-policy(--> Str:D) {
	return $!truncation-policy if $!truncation-policy.chars;
	&!parser.defined ?? 'fail' !! 'accept';
}

#|( Abort an in-flight response.

    Routes through C<Backend.cancel>, never C<Response.cancel> directly:
    closing the local supplier is only half of a cancel. KoboldCpp's
    override also fires C<POST /api/extra/abort> so the server stops
    generating, instead of draining a full completion into a socket
    nobody is reading any more. OpenAI-compatible backends
    (OpenAICommon, OpenRouter) terminate the HTTP exchange at the
    socket, which openai-compat servers treat as their abort signal —
    the server stops generating when the client disconnects. Mock
    degrades to the local close, which for it is the whole job.

    A backend whose C<cancel> throws must never leave the poll loop
    spinning on a live response, so the bare local close is the
    fallback. )
method !abort-response(LLM::Chat::Backend $backend, $response --> Nil) {
	$backend.cancel($response);
	CATCH { default { try $response.cancel } }
}

#|( Make one chat-completion round-trip against the given backend;
    always return a hash with C<response>, C<text>, C<success>,
    C<error>, C<error-class>, C<error-status>, C<elapsed-ms>.
    Never throws — callers rely on C<%call<success>> to branch (a
    caller-side cancel is reported as C<cancelled => True>; execute
    converts it to the typed throw AFTER firing telemetry).
    A caller-level deadline (C<$!timeout>) that fires during polling
    is classified as C<error-class => 'timeout'>; the C<is-cancelled>
    hook firing aborts the pending response the same way; a defined
    C<error-class>/C<error-status> on the underlying Response is
    copied verbatim. )
method !call-blocking(@messages, LLM::Chat::Backend $backend --> Hash) {
	my $t0 = now;
	my Str $text = Str;
	my Str $err  = Str;
	my Bool $success = False;
	my $response;
	my Str $error-class;
	my Int $error-status;
	my Bool $was-cancelled = False;
	try {
		$response = $backend.chat-completion(@messages);

		my Instant $deadline = now + $!timeout;
		my Bool $timed-out = False;
		until $response.is-done {
			if &!is-cancelled.defined && ?&!is-cancelled() {
				self!abort-response($backend, $response);
				$was-cancelled = True;
				last;
			}
			if now > $deadline {
				self!abort-response($backend, $response);
				$timed-out = True;
				last;
			}
			sleep 0.01;
		}

		if $was-cancelled {
			$err          = 'cancelled by caller mid-call';
			$error-class  = 'cancelled';
		}
		elsif $timed-out {
			$err          = "response timed out after {$!timeout}s";
			$error-class  = 'timeout';
		}
		elsif $response.is-success {
			$success = True;
			$text    = $response.msg;
		}
		else {
			my $raw = $response.err;
			my $raw-msg = $raw.defined
				?? ($raw ~~ Exception ?? exception-text($raw) !! ~$raw)
				!! 'unknown error';
			$err          = "LLM call failed: $raw-msg";
			$error-class  = $response.error-class;
			$error-status = $response.error-status;
			# A quit response can still carry text it emitted before
			# the failure — a stream cut off by the token budget is
			# the common case. Capture it so the partial generation is
			# available to callers instead of being dropped on the
			# floor; the failure classification is untouched.
			$text = $response.msg
				if $response.msg.defined && $response.msg.chars;
		}
		CATCH {
			default {
				$err         = "LLM::Data::Inference::Task: {exception-text($_)}";
				$error-class = 'unknown';
			}
		}
	}
	my Int $elapsed-ms = ((now - $t0) * 1000).Int;
	%(
		response     => $response,
		text         => $text,
		success      => $success,
		error        => $err,
		error-class  => $error-class,
		error-status => $error-status,
		elapsed-ms   => $elapsed-ms,
		cancelled    => $was-cancelled,
	);
}

#|( The empty-body rule, shared by the empty-completion branch and the
    thinking salvage so the two can never disagree: no text, or no
    characters, or — only once C<:empty-retries> has opted the Task
    into it — nothing but whitespace. )
method !empty-body(Str $text --> Bool:D) {
	!($text.defined && $text.chars && (!$!empty-retries || $text.trim.chars));
}

#|( Whether this round-trip's empty body is answered from its thinking:
    the option is on, the call succeeded, the body is empty by
    C<!empty-body>, the finish reason is 'stop' or absent, the Response
    carries no tool calls, and its C<reasoning-text> has characters. A
    'length' finish is excluded on purpose — the truncation rule owns
    it, and a think block the budget cut off is not an answer. )
method !takes-reasoning-as-response(%call, Str:D $finish --> Bool:D) {
	return False unless $!empty-response-uses-reasoning && %call<success>;
	my $response = %call<response>;
	return False without $response;
	return False unless self!empty-body(%call<text>);
	return False unless $finish eq '' || $finish eq 'stop';
	return False if $response.has-tool-calls;
	so ($response.reasoning-text // '').chars;
}

# Content-free diagnostics; absent provider values stay absent.
method !response-metadata($response --> Hash) {
	return %() unless $response.defined;
	my %metadata;
	if $response.msg.defined {
		%metadata<answer-chars> = $response.msg.chars;
		%metadata<answer-visible-chars> = $response.msg.trim.chars;
	}
	%metadata<reasoning-chars> = $response.reasoning-text.chars
		if $response.reasoning-text.defined;
	for <finish-reason prompt-tokens completion-tokens total-tokens> -> $field {
		my $value = $response."$field"();
		next unless $value.defined;
		next if $field eq 'finish-reason' && !$value.chars;
		%metadata{$field} = $value;
	}
	%metadata;
}

#|( Invoke the on-call-complete hook with a flattened payload.
    Shields the caller from hook exceptions — a broken telemetry
    sink must never interrupt the Task's retry loop. )
method !fire-telemetry(
	%call,
	:$attempt,
	:$backend-index,
	:$model-name,
	Bool:D :$used-reasoning-as-response = False,
) {
	return unless &!on-call-complete.defined;
	# Payload shape (which keys are always present, which are gated on
	# the provider actually reporting them) is the shared contract —
	# see LLM::Chat::Retry's telemetry-payload.
	my %payload = telemetry-payload(
		:$attempt,
		:$backend-index,
		:$model-name,
		latency-ms   => %call<elapsed-ms>,
		success      => %call<success>.Bool,
		stage        => 'network',
		error        => %call<error>,
		error-class  => %call<error-class>,
		error-status => %call<error-status>,
		response     => %call<response>,
	);
	%payload<finish-reason>:delete
		if %payload<finish-reason>.defined && !%payload<finish-reason>.chars;
	for self!response-metadata(%call<response>).kv -> $key, $value {
		%payload{$key} = $value;
	}
	# Always present, so a consumer reads it as a plain Bool: True only
	# when this row's empty body is being answered from its thinking
	# (see $.empty-response-uses-reasoning).
	%payload<used-reasoning-as-response> = $used-reasoning-as-response;
	try {
		&!on-call-complete(%payload);
		CATCH { default { note "telemetry hook: {exception-text($_)}" } }
	}
}

#|( Invoke the on-exhausted hook once, immediately before the chain
    throws C<Exhausted>. Shields the caller from hook exceptions —
    a broken hook must never suppress or alter the Exhausted throw
    that always follows. )
method !fire-exhausted(@attempt-records, Str:D $summary) {
	return unless &!on-exhausted.defined;
	my %payload = (
		stage    => 'exhausted',
		attempts => @attempt-records,
		summary  => $summary,
		backends => @!backends.elems,
	);
	try {
		&!on-exhausted(%payload);
		CATCH { default { note "exhausted hook: {exception-text($_)}" } }
	}
}
