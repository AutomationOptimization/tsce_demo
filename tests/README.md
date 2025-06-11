# Test Suite

All tests run without contacting external APIs. The `mock_tsce_chat` fixture in
`tests/conftest.py` replaces `tsce_agent_demo.tsce_chat.TSCEChat` with a simple
dummy implementation and sets a dummy `OPENAI_API_KEY`.

Use the fixture by adding `mock_tsce_chat` as an argument to your test. Most
unit tests depend on it implicitly and therefore do not require any real API
credentials.
