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
is aborted via C<Response.cancel>, exactly like the timeout path),
between backoff sleeps, and before parse re-rolls / backend advances.
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

		loop {
			# One check point covers the initial call, post-backoff
			# retries, parse re-rolls, and backend advances — every
			# path re-enters the loop top before touching the network.
			throw-cancelled if cancelled-now;
			$attempt++;
			my %call = self!call-blocking(@messages, $backend);
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

	my Str:D $summary =
		"LLM::Data::Inference::Task: all {@!backends.elems} backend(s) exhausted.\n"
		~ @errors.map({ "  $_" }).join("\n");
	self!fire-exhausted(@attempt-records, $summary);
	X::LLM::Data::Inference::Exhausted.new(
		attempts => @attempt-records,
		summary  => $summary,
	).throw;
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
				$response.cancel;
				$was-cancelled = True;
				last;
			}
			if now > $deadline {
				$response.cancel;
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
