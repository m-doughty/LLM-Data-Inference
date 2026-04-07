use LLM::Chat::Backend;
use LLM::Chat::Backend::Response;
use LLM::Chat::Conversation::Message;

unit class LLM::Data::Inference::Task;

has LLM::Chat::Backend $.backend is required;
has Str $.system-prompt;
has Str:D $.user-prompt is required;
has Int:D $.max-retries = 3;
has &.parser;
has Num:D $.timeout = 120e0;

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

	my Int:D $attempts = 0;
	my $last-error;

	loop {
		$attempts++;
		my Str:D $text = self!call-blocking(@messages);

		# No parser — return raw text
		return $text unless &!parser.defined;

		# Try parsing
		try {
			my $result = &!parser($text);
			return $result;
			CATCH {
				default {
					$last-error = $_;
					if $attempts >= $!max-retries {
						die "LLM::Data::Inference::Task: all {$!max-retries} attempts failed. "
							~ "Last error: {$last-error.message}";
					}
				}
			}
		}
	}
}

method !call-blocking(@messages --> Str:D) {
	my LLM::Chat::Backend::Response $response = $!backend.chat-completion(@messages);

	# Poll until done or timeout
	my Instant $deadline = now + $!timeout;
	until $response.is-done {
		if now > $deadline {
			$response.cancel;
			die "LLM::Data::Inference::Task: response timed out after {$!timeout}s";
		}
		sleep 0.01;
	}

	unless $response.is-success {
		die "LLM::Data::Inference::Task: LLM call failed: {$response.err // 'unknown error'}";
	}

	$response.msg;
}
