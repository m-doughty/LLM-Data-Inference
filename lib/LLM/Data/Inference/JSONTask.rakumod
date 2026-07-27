=begin pod

=head1 NAME

LLM::Data::Inference::JSONTask - Task that parses JSON output with schema checks

=head1 SYNOPSIS

=begin code :lang<raku>

use LLM::Data::Inference::JSONTask;

# Single backend (legacy)
my $task = LLM::Data::Inference::JSONTask.new(
    backend        => $backend,
    user-prompt    => '...',
    required-keys  => <name age>,
    validator      => -> %h { %h<age> > 0 },
);
my %parsed = $task.execute;

# Model-chain fallback — on malformed JSON / missing keys / validator
# failure, the underlying Task advances to the next backend.
my $task2 = LLM::Data::Inference::JSONTask.new(
    backends       => [$primary, $fallback],
    user-prompt    => '...',
    required-keys  => <name age>,
);

=end code

=head1 DESCRIPTION

Thin composition over L<LLM::Data::Inference::Task> that plugs in a
JSON parser + key-presence + custom-validator pipeline. All fallback
and retry semantics come from the underlying Task — see its Pod for
the full bucket classification. C<:parse-retries> threads through for
same-backend re-rolls on malformed output; C<:truncation-policy>
threads through too and, left at its default, resolves to C<'fail'>
here — a JSON task always has a parser, and a completion cut off by
C<max_tokens> can never parse, so the chain advances at once and
throws C<X::LLM::Data::Inference::Truncated> (an C<Exhausted>
subclass) if every backend truncates; C<:is-cancelled> threads
through for cooperative cancellation (the Task throws
C<X::LLM::Data::Inference::Cancelled> and aborts any in-flight
response as soon as the hook reports True); C<:on-exhausted> threads
through unchanged and fires exactly once, immediately before the
chain throws C<X::LLM::Data::Inference::Exhausted> — see Task's Pod
for the full payload shape.

C<:retry-feedback> threads through as well, and this is the class it
was built for: it makes those C<:parse-retries> re-rolls B<informed>,
appending one extra C<user> turn that quotes the rejection — a JSON
syntax error, a missing C<@.required-keys> entry, or whatever message
C<&.validator> died with. A semantic rejection ("quote 2 does not
appear verbatim in the passage") is otherwise re-produced verbatim on
every blind re-roll, so the item burns its whole budget failing
identically; with feedback the second attempt is a correction pass.
Each re-roll replaces the previous feedback, and only re-rolls against
the same backend are informed — see Task's "Retry feedback" Pod
section for the full rules.

Accepts either C<:$backend> (single) or C<:@backends> (chain); both
forms thread through to the inner Task unchanged.

=head2 JSON extraction

The extractor tolerates the chatter real models wrap around their
answer: C<< <think>…</think> >> reasoning blocks and markdown code
fences are stripped, then a string-aware balanced-bracket scan
collects every complete top-level JSON structure and picks the
longest one that actually parses (models that plan out loud often
emit a small throwaway object before the real answer). Only when no
complete structure parses does it fall back to the historical
first-bracket → last-bracket slice — so truncated completions still
fail with the established error messages.

=end pod

use JSON::Fast;
use LLM::Chat::Backend;
use LLM::Data::Inference::Task;

# Composition instead of inheritance — builds a Task with a JSON parser
unit class LLM::Data::Inference::JSONTask;

has LLM::Chat::Backend $.backend;
has LLM::Chat::Backend @.backends;
has Str $.system-prompt;
has Str:D $.user-prompt is required;
has Int:D $.max-retries = 3;
#| Same-backend re-roll budget for parse/validation failures —
#| threaded through to the inner Task (see its Pod for the trade-off).
has UInt:D $.parse-retries = 0;

#|( Give parse re-rolls the rejection reason, threaded through to the
    inner Task — see its "Retry feedback" Pod section. A JSON task is
    exactly the case the flag was built for: the failures it produces
    (JSON syntax, C<@.required-keys> misses, C<&.validator> verdicts)
    all come with a message that names what to fix, so an informed
    re-roll can converge where a blind one repeats itself. Each re-roll
    replaces the previous feedback; only re-rolls against the SAME
    backend are informed. Costs roughly a hundred extra prompt tokens
    on those attempts and nothing at all on the first. )
has Bool:D $.retry-feedback = False;

#|( Truncation policy, threaded through to the inner Task — see its
    "Truncated output" Pod section. Left at the default C<''> the
    inner Task derives C<'fail'>, because a JSONTask ALWAYS installs a
    parser: a completion cut off by C<max_tokens> is unparseable JSON
    by construction, so it advances to the next backend immediately
    instead of burning C<parse-retries> re-rolling the same
    over-budget request. Pass C<'accept'> to get the historical
    behaviour (hand the partial text to the parser and let it fail
    on its own terms). )
has Str:D $.truncation-policy where { $_ eq any('', 'fail', 'accept') } = '';

has Num:D $.timeout = 120e0;
has Str @.required-keys;
has &.validator;

#|( Optional telemetry hook threaded through to the inner Task.
    One fire per HTTP round-trip across the whole fallback chain. )
has &.on-call-complete;

#|( Optional cancellation hook threaded through to the inner Task —
    see L<LLM::Data::Inference::Task>'s cooperative-cancellation
    section. True → the Task aborts any in-flight response and throws
    C<X::LLM::Data::Inference::Cancelled>. )
has &.is-cancelled;

#|( Optional exhaustion hook threaded through to the inner Task —
    see L<LLM::Data::Inference::Task>'s telemetry section for the
    full payload shape (C<stage>, C<attempts>, C<summary>,
    C<backends>). Fires exactly once, immediately before the chain
    throws C<X::LLM::Data::Inference::Exhausted>. )
has &.on-exhausted;

submethod TWEAK() {
	if @!backends.elems == 0 {
		die "LLM::Data::Inference::JSONTask requires :backend or :backends"
			unless $!backend.defined;
		@!backends = ($!backend,);
	}
	$!backend = @!backends[0] without $!backend;
}

method execute(--> Any) {
	my @keys = @!required-keys;
	my &val = &!validator;

	my &json-parser = sub (Str:D $text) {
		my Str:D $json = extract-json($text);
		my $parsed = from-json($json);

		if $parsed ~~ Hash && @keys.elems > 0 {
			for @keys -> Str:D $key {
				die "Missing required key '$key' in JSON response"
					unless $parsed{$key}:exists;
			}
		}

		if &val.defined {
			die "Custom validation failed" unless &val($parsed);
		}

		$parsed;
	};

	LLM::Data::Inference::Task.new(
		:backends(@!backends),
		:system-prompt($!system-prompt),
		:$!user-prompt,
		:$!max-retries,
		:$!parse-retries,
		:$!retry-feedback,
		:$!truncation-policy,
		:$!timeout,
		:parser(&json-parser),
		:on-call-complete(&!on-call-complete),
		:is-cancelled(&!is-cancelled),
		:on-exhausted(&!on-exhausted),
	).execute;
}

#|( Strip non-JSON chatter that reliably poisons naive slicing:
    well-formed C<< <think>…</think> >> reasoning blocks and
    markdown code-fence marker lines (the fenced I<content> stays). )
sub strip-noise(Str:D $text --> Str:D) {
	my $t = $text;
	$t = $t.subst(/'<think>' .*? '</think>'/, ' ', :g);
	$t = $t.subst(/^^ \h* '```' \N* $$/, ' ', :g);
	$t;
}

#|( String-aware balanced scan: given the index of an opening C<{> or
    C<[>, walk forward tracking nesting depth while honouring JSON
    string literals (braces inside strings, escaped quotes). Returns
    the index of the closing bracket that returns to depth 0, or an
    undefined Int when the structure never closes. )
sub balanced-end(@chars, Int:D $start --> Int) {
	my Int $depth = 0;
	my Bool $in-str = False;
	my Bool $esc = False;
	loop (my $i = $start; $i < @chars.elems; $i++) {
		my $ch = @chars[$i];
		if $in-str {
			if $esc          { $esc = False }
			elsif $ch eq '\\' { $esc = True }
			elsif $ch eq '"' { $in-str = False }
			next;
		}
		given $ch {
			when '"'       { $in-str = True }
			when '{' | '[' { $depth++ }
			when '}' | ']' {
				$depth--;
				return $i if $depth == 0;
			}
		}
	}
	Int;
}

sub extract-json(Str:D $text --> Str:D) {
	# Preferred path: strip reasoning/fence noise, then collect every
	# COMPLETE top-level JSON structure via a balanced scan (a stray
	# brace in leftover prose just fails its scan and is skipped).
	# Of the complete candidates, take the longest that from-json
	# actually accepts — models that "plan out loud" often emit a
	# small throwaway object before the real answer, and the real
	# answer dwarfs it; ties go to the later one.
	my $work = strip-noise($text);
	my @chars = $work.comb;
	my @candidates;
	my Int $pos = 0;
	while $pos < @chars.elems {
		my $s = @chars[$pos..*].first({ $_ eq '{' || $_ eq '[' }, :k);
		last without $s;
		$s += $pos;
		with balanced-end(@chars, $s) -> $e {
			@candidates.push: %( start => $s, end => $e, len => $e - $s + 1 );
			$pos = $e + 1;
		} else {
			$pos = $s + 1;
		}
	}
	for @candidates.sort({ $^b<len> <=> $^a<len> || $^b<start> <=> $^a<start> }) -> %c {
		my Str $slice = @chars[%c<start> .. %c<end>].join;
		return $slice if try { from-json($slice); True };
	}

	# Fallback: the historical first-bracket → last-bracket slice, so
	# hopeless responses keep producing the established error messages.
	legacy-slice($text);
}

sub legacy-slice(Str:D $text --> Str:D) {
	# Pick whichever structure starts earlier in the text. Previous
	# version always preferred `{` over `[`, which broke array
	# responses wrapped in prose: the first `{` inside the array
	# would be grabbed, returning only the first object and failing
	# with a "trailing text" error.
	my $obj-start = $text.index('{');
	my $arr-start = $text.index('[');

	my $use-array = do {
		if !$obj-start.defined && $arr-start.defined {
			True;
		}
		elsif !$arr-start.defined && $obj-start.defined {
			False;
		}
		elsif $obj-start.defined && $arr-start.defined {
			$arr-start < $obj-start;
		}
		else {
			die "No JSON object or array found in response";
		}
	};

	if $use-array {
		my $arr-end = $text.rindex(']');
		if $arr-end.defined && $arr-end > $arr-start {
			return $text.substr($arr-start, $arr-end - $arr-start + 1);
		}
		die "Found '[' but no matching ']' in response";
	}
	else {
		my $obj-end = $text.rindex('}');
		if $obj-end.defined && $obj-end > $obj-start {
			return $text.substr($obj-start, $obj-end - $obj-start + 1);
		}
		die "Found '{' but no matching '}' in response";
	}
}
