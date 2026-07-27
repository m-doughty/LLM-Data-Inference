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
        # Truncated IS an Exhausted, so it must come FIRST — a `when`
        # chain takes the first matching arm, and the Exhausted arm
        # below would otherwise swallow it.
        when X::LLM::Data::Inference::Truncated {
            note "every attempt hit the token budget:\n{.message}";
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
C<Truncated>, and C<:on-exhausted> fires identically for both.

=end pod

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
    unaffected. )
class X::LLM::Data::Inference::Exhausted is Exception is export {
	has @.attempts;
	has Str:D $.summary is required;

	method message(--> Str) { $!summary }
}

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
    garbage" can, by matching it BEFORE the C<Exhausted> arm. )
class X::LLM::Data::Inference::Truncated
	is X::LLM::Data::Inference::Exhausted is export { }

#|( Thrown when the Task's C<is-cancelled> hook reports the caller
    withdrew mid-execute. C<attempts> matches C<Exhausted>'s shape
    (whatever failures were recorded before the cancel landed —
    possibly none). Deliberately NOT a subclass of C<Exhausted>:
    handlers that treat exhaustion as a model failure must not
    accidentally swallow a user cancel. )
class X::LLM::Data::Inference::Cancelled is Exception is export {
	has @.attempts;

	method message(--> Str) {
		'LLM::Data::Inference::Task: cancelled by caller'
		~ (@!attempts
			?? " after {@!attempts.elems} recorded attempt(s)"
			!! '');
	}
}
