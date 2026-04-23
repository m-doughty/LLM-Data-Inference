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
(different model may produce parseable output). The Task does NOT retry
the same backend on parser failure — the previous per-backend retry
loop produced identical malformed output enough times in practice that
burning budget on same-model parse retries was net-negative.

=head2 Telemetry

Every HTTP round-trip fires C<&on-call-complete> (if provided), with
the attempt number, model name, backend index (position in the chain),
latency, success flag, usage data, and error details. Parser failures
do not fire separately — they're classified from the preceding
telemetry row.

=head1 BACKWARD COMPATIBILITY

Callers passing C<:$backend> only see no behaviour change beyond the
new retry policy. The C<$.backend> accessor remains available and
always points at the first backend in the chain; legacy consumers that
read it (e.g. to label log messages with the model) keep working.

=end pod

use LLM::Chat::Backend;
use LLM::Chat::Backend::Response;
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
      * prompt-tokens / completion-tokens / total-tokens / cost /
        model-used / provider-id / finish-reason — presence-gated,
        lifted off the Response when the provider supplies them. )
has &.on-call-complete;

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

	for @!backends.kv -> $i, $backend {
		my $same-retries-left = $!max-retries > 0 ?? $!max-retries - 1 !! 0;
		my $model-name = (try { $backend.model }) // 'unknown';

		loop {
			$attempt++;
			my %call = self!call-blocking(@messages, $backend);
			self!fire-telemetry(
				%call,
				:$attempt,
				:backend-index($i),
				:$model-name,
			);

			# Network-layer failure
			unless %call<success> {
				my $bucket = self.classify-error(
					error-class  => %call<error-class>,
					error-status => %call<error-status>,
				);
				@errors.push:
					"[backend $i $model-name] "
					~ "{%call<error-class> // 'unknown'}"
					~ (%call<error-status>.defined ?? " {%call<error-status>}" !! '')
					~ ": {%call<error>}";

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
							# one); capped at 30 s.
							my $retry-n = $!max-retries - $same-retries-left;
							my Num $wait = min(
								(2 ** ($retry-n - 1)).Num + rand * 0.5,
								30e0,
							);
							$same-retries-left--;
							sleep $wait;
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
				@errors.push:
					"[backend $i $model-name] empty response body";
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

			@errors.push:
				"[backend $i $model-name] parser: $parse-error";
			last;  # advance
		}
	}

	die "LLM::Data::Inference::Task: all {@!backends.elems} backend(s) exhausted.\n"
		~ @errors.map({ "  $_" }).join("\n");
}

#|( Make one chat-completion round-trip against the given backend;
    always return a hash with C<response>, C<text>, C<success>,
    C<error>, C<error-class>, C<error-status>, C<elapsed-ms>.
    Never throws — callers rely on C<%call<success>> to branch.
    A caller-level deadline (C<$!timeout>) that fires during polling
    is classified as C<error-class => 'timeout'>; a defined
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
	try {
		$response = $backend.chat-completion(@messages);

		my Instant $deadline = now + $!timeout;
		my Bool $timed-out = False;
		until $response.is-done {
			if now > $deadline {
				$response.cancel;
				$timed-out = True;
				last;
			}
			sleep 0.01;
		}

		if $timed-out {
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
		%payload<prompt-tokens>     = $r.prompt-tokens     if $r.prompt-tokens.defined;
		%payload<completion-tokens> = $r.completion-tokens if $r.completion-tokens.defined;
		%payload<total-tokens>      = $r.total-tokens      if $r.total-tokens.defined;
		%payload<cost>              = $r.cost              if $r.cost.defined;
		%payload<model-used>        = $r.model-used        if $r.model-used.defined;
		%payload<provider-id>       = $r.provider-id       if $r.provider-id.defined;
		%payload<finish-reason>     = $r.finish-reason     if $r.finish-reason.defined;
	}
	try {
		&!on-call-complete(%payload);
		CATCH { default { note "telemetry hook: {.message}" } }
	}
}
