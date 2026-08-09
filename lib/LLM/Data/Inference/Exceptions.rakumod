=begin pod

=head1 NAME

LLM::Data::Inference::Exceptions - typed failures thrown by the inference chain

=head1 DESCRIPTION

C<X::LLM::Data::Inference::Exhausted> is thrown by
L<LLM::Data::Inference::Task> when every backend in the chain has been
tried without producing a usable result. Its C<message> is the same
human-readable summary the module has always died with, so string-based
callers keep working — but the exception additionally carries the
per-attempt failure records, including the B<raw model output> for
parser failures, so callers can show the user exactly what the model
produced instead of only "parser: unexpected ']'".

C<X::LLM::Data::Inference::Truncated> is a B<subclass> of
C<Exhausted>, thrown in its place when at least one attempt in the
exhausted chain died because the model ran out of completion budget
(the provider reported C<finish_reason> C<'length'>) rather than
because it produced something unusable. Being a subclass is the whole
point: every existing C<Exhausted> handler — DLQ routing, CLI error
reporting, C<:on-exhausted> hooks — keeps catching it unchanged, while
callers that care can add a narrower arm to say "this one is fixable:
raise the budget" instead of "the model can't do this". It carries no
extra attributes; the lever is named in the C<summary>, and the
truncated attempt records carry their partial output as C<raw-text>.
Only a Task with C<truncation-policy> C<'fail'> can throw it — see
L<LLM::Data::Inference::Task>.

C<X::LLM::Data::Inference::TimedOut> is likewise a B<subclass> of
C<Exhausted>, thrown in its place when at least one attempt in the
exhausted chain died on a response deadline — the Task's own
C<$.timeout> firing mid-poll, or a backend reporting error class
C<'timeout'>. Same rationale as C<Truncated>: exhaustion by deadline
IS a chain failure and must keep flowing to every existing
C<Exhausted> handler untouched, while callers who want to distinguish
"fixable by raising the deadline" from "the model can't do this" can
match it B<before> the C<Exhausted> arm. The lever is named in the
C<summary>, which quotes the deadline that was hit.

When a chain saw B<both> a truncation and a timeout, C<Truncated>
wins: it is the more specific diagnosis and, unlike a timeout, it is
deterministic — see the advice contract below.

C<X::LLM::Data::Inference::Cancelled> is thrown by the same Task when
its C<is-cancelled> hook reports that the caller no longer wants the
result — before a round-trip, between retries, or mid-call (the
in-flight response is aborted first). It carries the same per-attempt
records as C<Exhausted> so anything already produced stays
inspectable. Catch it separately from C<Exhausted>: cancellation is
the caller's own intent, not a failure of the chain.

=head1 EXAMPLES

=begin code :lang<raku>

use LLM::Data::Inference::Task;
use LLM::Data::Inference::Exceptions;

my $task = LLM::Data::Inference::Task.new(
    :@backends, :$user-prompt, :&parser, parse-retries => 2,
    is-cancelled => -> { $job.cancelled },
);

my $result = do {
    CATCH {
        when X::LLM::Data::Inference::Cancelled {
            # The caller asked for this — clean up quietly.
            %();
        }
        # Truncated and TimedOut ARE Exhausteds, so they must come
        # FIRST — a `when` chain takes the first matching arm, and the
        # Exhausted arm below would otherwise swallow them.
        when X::LLM::Data::Inference::Truncated {
            note "every attempt hit the token budget:\n{.message}";
            %();
        }
        when X::LLM::Data::Inference::TimedOut {
            note "the chain ran out of time:\n{.message}";
            %();
        }
        when X::LLM::Data::Inference::Exhausted {
            for .attempts.grep(*.<raw-text>.defined) -> %a {
                note "model %a<model> said:\n%a<raw-text>";
            }
            %();
        }
    }
    $task.execute;
};

=end code

Callers that do not care about the distinction need no change at all:
a bare C<when X::LLM::Data::Inference::Exhausted> still catches a
C<Truncated> or a C<TimedOut>, and C<:on-exhausted> fires identically
for all three.

=head1 RELATIONSHIP TO C<LLM::Chat::Retry::Exceptions>

Since 0.9.0 these four types are thin B<subclasses> of the shared
retry exceptions in L<LLM::Chat::Retry::Exceptions>, which is where
the retry/fallback policy this distribution pioneered now lives so a
second executor can share it:

=begin code :lang<text>

X::LLM::Chat::Retry::Exhausted
├── X::LLM::Data::Inference::Exhausted
│   ├── X::LLM::Data::Inference::Truncated  (also X::LLM::Chat::Retry::Truncated)
│   └── X::LLM::Data::Inference::TimedOut   (also X::LLM::Chat::Retry::TimedOut)
├── X::LLM::Chat::Retry::Truncated
└── X::LLM::Chat::Retry::TimedOut

X::LLM::Chat::Retry::Cancelled
└── X::LLM::Data::Inference::Cancelled

=end code

The consequence is that B<both> hierarchies match, so nothing has to
choose:

=begin code :lang<raku>

my $ex = ... ;  # thrown by a Task
$ex ~~ X::LLM::Data::Inference::Exhausted;   # True — every existing handler
$ex ~~ X::LLM::Chat::Retry::Exhausted;       # True — a generic retry handler

=end code

Attributes (C<@.attempts>, C<$.summary>), C<.message> strings, the
attempt-record shape and the C<item-retryable> advice are all
unchanged; the diamond exists purely so a caller that handles chains
from several executors can write one C<when> arm against the shared
types, while a caller that only knows this distribution never notices.

Two notes for anyone subclassing further: match the more specific type
B<first> in a C<when> chain, and use the public accessor C<@.attempts>
rather than C<@!attempts> when overriding a method — the attribute now
lives in the C<LLM::Chat> parent and is not visible to a subclass's
private-attribute syntax.

=head1 THE C<item-retryable> ADVICE CONTRACT

The contract's canonical documentation moved to
L<LLM::Chat::Retry::Exceptions> along with the types themselves. In
brief: an exception may B<advise> an orchestration layer that re-runs
failed work (C<LLM::Data::Pipeline>'s item engine, a job queue, a
supervisor loop) that a blind re-roll cannot possibly help, by
exposing

=begin code :lang<raku>

method item-retryable(--> Bool:D) { False }

=end code

which the layer reads as C<< $ex.?item-retryable // True >> — absent
means retryable, so every exception that has never heard of the
contract keeps its full attempt budget, and neither side imports the
other's types.

In this distribution only C<Truncated> declares it, as C<False>: a
completion cut off by C<max_tokens> will be cut off at exactly the
same place on the next blind re-roll, because C<max_tokens> is
per-backend C<Settings> state that a re-run cannot change. C<TimedOut>
deliberately does B<not> — a deadline miss is genuinely transient (a
slow provider, a queued request, an upstream hiccup), so it keeps the
full item budget. Plain C<Exhausted> abstains for the same reason, and
nothing in this distribution reads the method: it is advice, not
control flow.

=end pod

use LLM::Chat::Retry::Exceptions;

unit module LLM::Data::Inference::Exceptions;

#|( Thrown when the whole backend chain is exhausted. C<attempts>
    holds one Hash per recorded failure, in order:
      * backend-index — 0-based position in the chain
      * model         — model name (or 'unknown')
      * error         — the failure description
      * raw-text      — the full model output, present only for
                        parser failures (network-level failures have
                        no body to keep)
    C<message> returns the same aggregate summary string the Task
    used to C<die> with, so existing C<.message>-matching callers are
    unaffected.

    Since 0.9.0 a subclass of C<X::LLM::Chat::Retry::Exhausted>, which
    is where the attributes and C<.message> now live — the shape and
    behaviour are identical, and instances match BOTH hierarchies so
    generic retry handlers and inference-specific ones both fire.

    Deliberately does NOT define C<item-retryable>: the base
    exhaustion carries no evidence that a re-run is pointless, and
    the advice contract reads an absent method as "retryable" (see
    the module Pod). Subclasses that DO have that evidence declare
    it — C<Truncated> does. )
class X::LLM::Data::Inference::Exhausted
	is X::LLM::Chat::Retry::Exhausted is export { }

#|( Thrown INSTEAD of C<Exhausted> when at least one attempt in the
    exhausted chain was cut off by the completion budget — the
    provider reported C<finish_reason> 'length' and the Task's
    C<truncation-policy> is 'fail'. Same shape as C<Exhausted> (no
    extra attributes): the truncation is described in C<summary>,
    which also names the levers (raise C<max_tokens>, add a
    larger-budget fallback backend, or shrink the request), and each
    truncated attempt record keeps whatever partial output arrived as
    C<raw-text>.

    Deliberately a SUBCLASS of C<Exhausted>, unlike C<Cancelled>:
    truncation IS a chain failure and belongs in the dead-letter
    queue, so every existing handler must keep working untouched.
    The subclass only exists so callers who want to distinguish
    "fixable by raising the budget" from "the model produced
    garbage" can, by matching it BEFORE the C<Exhausted> arm.

    Wins over C<TimedOut> when a chain saw both: it is the more
    specific diagnosis, and it is the deterministic one.

    Also a subclass of C<X::LLM::Chat::Retry::Truncated>, which is
    where its C<item-retryable> False now comes from — the diamond is
    deliberate, so this type matches the inference hierarchy AND the
    shared one. )
class X::LLM::Data::Inference::Truncated
	is X::LLM::Data::Inference::Exhausted
	is X::LLM::Chat::Retry::Truncated is export { }

#|( Thrown INSTEAD of C<Exhausted> when at least one attempt in the
    exhausted chain died on a response deadline — the Task's own
    C<$.timeout> firing while it polled the in-flight response (the
    pending call is aborted through C<Backend.cancel> first), or a
    backend reporting error class 'timeout' itself. Same shape as
    C<Exhausted> (no extra attributes); the deadline that was hit and
    its levers are named in C<summary>.

    A SUBCLASS of C<Exhausted> for exactly the reasons C<Truncated>
    is one: a chain that ran out of time is a chain failure, belongs
    in the dead-letter queue, and must keep reaching every handler
    that already exists. The subclass earns its keep by letting a
    caller say "raise the deadline / use a faster backend" instead of
    "the model can't do this" — match it BEFORE the C<Exhausted> arm.

    Deliberately does NOT define C<item-retryable>. Unlike a
    truncation, a deadline miss is genuinely transient: the same
    request against the same chain a minute later routes to a
    different upstream provider, or simply is not queued behind
    someone else's batch. It keeps the full item-retry budget.

    Also a subclass of C<X::LLM::Chat::Retry::TimedOut>, so it
    matches the shared hierarchy too. )
class X::LLM::Data::Inference::TimedOut
	is X::LLM::Data::Inference::Exhausted
	is X::LLM::Chat::Retry::TimedOut is export { }

#|( Thrown when the Task's C<is-cancelled> hook reports the caller
    withdrew mid-execute. C<attempts> matches C<Exhausted>'s shape
    (whatever failures were recorded before the cancel landed —
    possibly none). Deliberately NOT a subclass of C<Exhausted>:
    handlers that treat exhaustion as a model failure must not
    accidentally swallow a user cancel.

    A subclass of C<X::LLM::Chat::Retry::Cancelled> (which is likewise
    not an C<Exhausted>), overriding only C<.message> so it keeps
    naming this Task by its own name — the string is byte-identical
    to what it has always been. Note C<@.attempts>, not C<@!attempts>:
    the attribute lives in the parent class now. )
class X::LLM::Data::Inference::Cancelled
	is X::LLM::Chat::Retry::Cancelled is export {

	method message(--> Str) {
		'LLM::Data::Inference::Task: cancelled by caller'
		~ (@.attempts
			?? " after {@.attempts.elems} recorded attempt(s)"
			!! '');
	}
}
