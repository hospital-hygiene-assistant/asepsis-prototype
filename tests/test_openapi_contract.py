"""Generated review-interface probes must fail as data, never as server errors."""

import schemathesis
from hypothesis import given, settings

import server


schema = schemathesis.openapi.from_asgi("/openapi.json", server.app)
get_review = schema["/api/ingest/reviews/{session_id}"]["get"]


@settings(max_examples=25, deadline=None)
@given(case=get_review.as_strategy())
def test_generated_review_identifiers_never_cause_an_internal_error(case):
    response = case.call()
    assert response.status_code < 500
