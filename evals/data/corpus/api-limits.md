# Fleet API rate limits

The Fleet API lets customers control robots and read telemetry programmatically. Requests are authenticated with a project token sent in the Authorization header.

The Starter plan allows 60 requests per minute per project. The Pro plan allows 600 requests per minute, and the Enterprise plan allows 6000 requests per minute with burst capacity. When a client exceeds its limit, the API answers with HTTP status 429 and a Retry-After header that says how many seconds to wait.

Telemetry streams over WebSocket do not count toward the request limit, but each project may keep at most 50 open telemetry streams at a time.
