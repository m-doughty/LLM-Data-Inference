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

=head2 Telemetry

Every HTTP round-trip fires C<&on-call-complete> (if provided), with
the attempt number, model name, backend index (position in the chain),
latency, success flag, usage data, and error details. Parser failures
do not fire separately — they're classified from the preceding
telemetry row.

C<&on-exhausted> (if provided) fires exactly once, on the execute
thread, immediately before the chain throws
C<X::LLM::Data::Inference::Exhausted> — never on success, and never
on the C<Cancelled> path. The payload is C<stage =E<gt> 'exhausted'>,
C<attempts> (the same C<@attempt-records> array the thrown exception
carries — including C<raw-text> for parse failures), C<summary> (the
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
C<POST /api/extra/abort> alongside the local stream close, so a
cancelled Task stops the generation instead of only looking away from
it. Backends with no upstream abort endpoint degrade to a local close.
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
    caller in normal use. )
method classify-error(
	Str :$error-class,
	Int :$error-status,
	Bool :$parser-failed = False,
	--> Str
) {
	return 'advance' if $parser-failed;
	given $error-class // 'unknown' {
		when 'http' {
			given $error-status // 0 {
				when 400 | 401 | 402 | 403 | 404 { return 'abort' }
				when 429                         { return 'advance' }
				when 500..599                    { return 'retry-same' }
				default                          { return 'advance' }
			}
		}
		when 'timeout'    { return 'advance' }
		when 'connection' { return 'retry-same' }
		when 'response'   { return 'advance' }
		default           { return 'retry-same' }
	}
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
	my @errors;
	my @attempt-records;
	my Str:D $truncation-policy = self!effective-truncation-policy;
	# Sticky across the whole chain: one truncated attempt anywhere is
	# enough to make the exhaustion Truncated-typed and to earn the
	# summary its "raise the budget" line.
	my Bool $saw-truncation = False;
	sub record-failure(Int $backend-index, Str $model, Str $error, Str :$raw-text) {
		@errors.push: $error;
		my %rec = backend-index => $backend-index, model => $model,
			error => $error;
		%rec<raw-text> = $raw-text if $raw-text.defined;
		@attempt-records.push: %rec;
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
			self!fire-telemetry(
				%call,
				:$attempt,
				:backend-index($i),
				:$model-name,
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
				record-failure($i, $model-name,
					"[backend $i $model-name] "
					~ "{%call<error-class> // 'unknown'}"
					~ (%call<error-status>.defined ?? " {%call<error-status>}" !! '')
					~ ": {%call<error>}");

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
							# cancel doesn't wait out the full backoff.
							my $retry-n = $!max-retries - $same-retries-left;
							my Num $wait = min(
								(2 ** ($retry-n - 1)).Num + rand * 0.5,
								30e0,
							);
							$same-retries-left--;
							my Num $slept = 0e0;
							while $slept < $wait {
								throw-cancelled if cancelled-now;
								my Num $chunk = min(0.25e0, $wait - $slept);
								sleep $chunk;
								$slept += $chunk;
							}
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
			my Str $finish = ((%call<response>.defined
				?? %call<response>.finish-reason
				!! Str) // '');
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
					raw-text => $partial);
				# Advance WITHOUT touching $parse-retries-left: the
				# same backend re-rolling the same over-budget request
				# will overrun again at the same max_tokens, while the
				# next backend may carry a bigger budget.
				last;
			}

			unless $text.defined && $text.chars {
				record-failure($i, $model-name,
					"[backend $i $model-name] empty response body");
				last;  # advance
			}

			# No parser — raw text is the result.
			return $text unless &!parser.defined;

			my $parsed;
			my $parse-failed = False;
			my $parse-error;
			try {
				$parsed = &!parser($text);
				CATCH {
					default {
						$parse-failed = True;
						$parse-error  = ~$_.message;
					}
				}
			}
			return $parsed unless $parse-failed;

			record-failure($i, $model-name,
				"[backend $i $model-name] parser: $parse-error",
				raw-text => $text);
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

	# The lever line only appears when a truncation was actually
	# recorded, so summaries of ordinary failures are byte-identical
	# to what they always were.
	my Str:D $lever = $saw-truncation
		?? "\n  At least one attempt was cut off by the completion "
			~ "budget: raise that backend's max_tokens completion "
			~ "budget, add a fallback backend with a larger one, or "
			~ "shrink the request."
		!! '';
	my Str:D $summary =
		"LLM::Data::Inference::Task: all {@!backends.elems} backend(s) exhausted.\n"
		~ @errors.map({ "  $_" }).join("\n")
		~ $lever;
	# Fires identically for both exception types — a DLQ sink sees the
	# same payload shape it always has, with the lever line included.
	self!fire-exhausted(@attempt-records, $summary);
	my $ex-type = $saw-truncation
		?? X::LLM::Data::Inference::Truncated
		!! X::LLM::Data::Inference::Exhausted;
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
    nobody is reading any more. Backends with no upstream abort
    (OpenAICommon, OpenRouter, Mock) degrade to exactly the local close
    this used to do, so the change is free for them.

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
				?? ($raw ~~ Exception ?? $raw.message !! ~$raw)
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
				$err         = "LLM::Data::Inference::Task: {$_.message}";
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

#|( Invoke the on-call-complete hook with a flattened payload.
    Shields the caller from hook exceptions — a broken telemetry
    sink must never interrupt the Task's retry loop. )
method !fire-telemetry(
	%call,
	:$attempt,
	:$backend-index,
	:$model-name,
) {
	return unless &!on-call-complete.defined;
	my %payload = (
		attempt       => $attempt,
		backend-index => $backend-index,
		model-name    => $model-name,
		latency-ms    => %call<elapsed-ms>,
		success       => %call<success>.Bool,
		stage         => 'network',
		error         => (%call<error>.defined && %call<error>.chars) ?? %call<error> !! Str,
		error-class   => %call<error-class>,
		error-status  => %call<error-status>,
	);
	my $r = %call<response>;
	if $r.defined {
		# OAI-spec usage — present on any LLM::Chat::Backend::Response.
		%payload<prompt-tokens>     = $r.prompt-tokens     if $r.prompt-tokens.defined;
		%payload<completion-tokens> = $r.completion-tokens if $r.completion-tokens.defined;
		%payload<total-tokens>      = $r.total-tokens      if $r.total-tokens.defined;
		%payload<model-used>        = $r.model-used        if $r.model-used.defined;
		%payload<finish-reason>     = $r.finish-reason     if $r.finish-reason.defined;
		# Provider-specific extras — only on Response subclasses that
		# expose them (e.g. LLM::Chat::Backend::Response::OpenRouter).
		# `.?` returns Nil for Responses without the accessor, so the
		# presence-gated contract below holds for any backend.
		%payload<cost>           = $r.?cost           if $r.?cost.defined;
		%payload<generation-id>  = $r.?generation-id  if $r.?generation-id.defined;
		%payload<provider-name>  = $r.?provider-name  if $r.?provider-name.defined;
		%payload<is-byok>        = $r.?is-byok        if $r.?is-byok.defined;
	}
	try {
		&!on-call-complete(%payload);
		CATCH { default { note "telemetry hook: {.message}" } }
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
		CATCH { default { note "exhausted hook: {.message}" } }
	}
}
