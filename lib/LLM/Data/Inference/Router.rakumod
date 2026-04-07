use LLM::Chat::Backend;
use Roaring::Tags;
use CRoaring;

unit class LLM::Data::Inference::Router;

has LLM::Chat::Backend $.default-backend is required;
has @!routes;  # List of { query => Str, backend => Backend }

method add-route(Str:D $query, LLM::Chat::Backend:D $backend --> Nil) {
	@!routes.push: %(:$query, :$backend);
}

method select-backend(Roaring::Tags:D $tags, Int:D $doc-id --> LLM::Chat::Backend:D) {
	for @!routes -> %route {
		my CRoaring $matches = $tags.search(%route<query>);
		my Bool:D $hit = $matches.contains($doc-id);
		$matches.dispose;
		return %route<backend> if $hit;
	}
	$!default-backend;
}
