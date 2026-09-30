# Playwright Fixtures

Playwright fixtures provide reusable setup and teardown for tests.

A fixture can create and configure objects required by tests, such as browser
pages, API clients, test data, or authenticated sessions.

Fixtures can be shared across multiple tests and can control the lifecycle of
the objects they create.

Playwright provides built-in fixtures such as `page`, `browser`, `context`, and
`request`.

Custom fixtures can be created when a project requires additional test
infrastructure.
