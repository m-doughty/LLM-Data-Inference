unit class LLM::Data::Inference::PromptBuilder;

has Str:D $.template is required;

method render(%vars --> Str:D) {
	my Str:D $result = $!template;
	# Find all {{key}} placeholders and replace
	$result = $result.subst(/ '{{' (<-[}]>+) '}}' /, -> $/ {
		my Str:D $key = ~$0;
		die "LLM::Data::Inference::PromptBuilder: missing variable '$key'"
			unless %vars{$key}:exists;
		~%vars{$key};
	}, :g);
	$result;
}
