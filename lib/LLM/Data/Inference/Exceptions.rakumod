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
