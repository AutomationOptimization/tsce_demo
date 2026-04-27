# HDA Task Suite Spec

This suite is a deterministic, black-box characterization layer for targeted HDA research. It defines task contracts and scorer dimensions only; it does not define any HDA search, selection, or optimization policy.

## Record Shape

Materialized rows use the existing JSONL shape:

```json
{"id":1,"prompt":"...","kind":"...","truth":"...","meta":{}}
```

`meta.generator` is `targeted_task_suite`. Stable metadata also includes `suite_version`, `subtype`, `difficulty`, `template_id`, and `task_kind_request` when sampled through `sample_task_records`.

## `problem_json`

`problem_json` expands to these six scoreable subtypes:

- `arithmetic_json`
- `calendar_json`
- `logic_json`
- `string_transform_json`
- `unit_conversion_json`
- `reading_comprehension_json`

Each prompt requires the exact JSON output contract:

```json
{"answer":""}
```

The scorer accepts only a JSON object whose keys are exactly `answer` for full credit. Answer comparison normalizes leading/trailing whitespace, repeated internal whitespace, case, numeric thousands separators, integer-looking floats, and one trailing period. The scored dimensions are:

- `json_valid`: a JSON object can be parsed.
- `schema_exact`: the parsed object has exactly one key, `answer`.
- `answer_correct`: normalized `answer` equals normalized truth.
- `no_extra_text`: the reply is only the JSON object.
- `full_pass`: all dimensions pass.

## `em_dash_removal`

The prompt asks the model to rewrite a short post to remove em dashes while preserving the rest of the text. Allowed replacements for each em dash are:

- removal with surrounding spacing normalized
- comma
- period
- colon
- semicolon

The scored dimensions are:

- `no_em_dash`: the reply contains no em dash character.
- `no_extra_text`: the reply contains only the rewritten post content.
- `minimal_edit`: every em dash was replaced by an allowed replacement and nothing else changed.
- `content_preserved`: word sequence matches the source after ignoring the em dash.
- `full_pass`: all dimensions pass.

## `exact_repeat`

The prompt asks for one line repeated an exact number of times, one per line, with no added text. Truth is an object with `line` and `count`.

The scored dimensions are:

- `exact_line_count`: the reply has exactly the requested number of lines.
- `exact_line_content`: every reply line exactly matches the requested line.
- `no_extra_text`: both line count and line content are exact.
- `full_pass`: all dimensions pass.

## Known Ambiguities

- `em_dash_removal`: punctuation choice is intentionally flexible among the allowed replacement set, so multiple fully passing rewrites can exist.
- `problem_json`: capitalization, spacing, integer formatting, and one trailing period are normalized for `answer`; other schema or extra-text differences are not normalized.
- `exact_repeat`: trailing final newlines are ignored, but trailing spaces on repeated lines are content differences.
- Evidence interpretation: partial dimension gains can be useful diagnostic evidence, but they are not equivalent to `full_pass` unless the analysis explicitly treats dimensions separately.
