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
the full bucket classification.

Accepts either C<:$backend> (single) or C<:@backends> (chain); both
forms thread through to the inner Task unchanged.

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
has Num:D $.timeout = 120e0;
has Str @.required-keys;
has &.validator;

#|( Optional telemetry hook threaded through to the inner Task.
    One fire per HTTP round-trip across the whole fallback chain. )
has &.on-call-complete;

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
		:$!timeout,
		:parser(&json-parser),
		:on-call-complete(&!on-call-complete),
	).execute;
}

sub extract-json(Str:D $text --> Str:D) {
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
