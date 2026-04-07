use JSON::Fast;
use LLM::Chat::Backend;
use LLM::Data::Inference::Task;

# Composition instead of inheritance — builds a Task with a JSON parser
unit class LLM::Data::Inference::JSONTask;

has LLM::Chat::Backend $.backend is required;
has Str $.system-prompt;
has Str:D $.user-prompt is required;
has Int:D $.max-retries = 3;
has Num:D $.timeout = 120e0;
has Str @.required-keys;
has &.validator;

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
		:$!backend,
		:system-prompt($!system-prompt),
		:$!user-prompt,
		:$!max-retries,
		:$!timeout,
		:parser(&json-parser),
	).execute;
}

sub extract-json(Str:D $text --> Str:D) {
	my Int $obj-start = $text.index('{');
	if $obj-start.defined {
		my Int $obj-end = $text.rindex('}');
		if $obj-end.defined && $obj-end > $obj-start {
			return $text.substr($obj-start, $obj-end - $obj-start + 1);
		}
	}

	my Int $arr-start = $text.index('[');
	if $arr-start.defined {
		my Int $arr-end = $text.rindex(']');
		if $arr-end.defined && $arr-end > $arr-start {
			return $text.substr($arr-start, $arr-end - $arr-start + 1);
		}
	}

	die "No JSON object or array found in response";
}
