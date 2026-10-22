# Error-message audit prompt

You are auditing the messages kpnn2 shows its callers: every
error, warning, and internal error built in the modules in
scope. Callers are people and the coding agents they work
with. Both read the message and must know what went wrong and
what to do next.

Tests already enforce the message format: a headline, then
optional `Got`, `Expected`, `Did you mean`, and `Fix` lines in
that order, one line each. You judge what tests cannot:

1. **Accuracy.** The headline and every field say what the
   code checks and what the caller passed.
2. **Usefulness.** A caller who knows their data and the
   public docs, but not kpnn2's internals, can act on the
   message.
3. **Field choice.** The message has the fields that help and
   no field that only fills a slot.
4. **Consistency.** The same rule gets the same words
   everywhere.

## Scope

The user names the modules to audit, for example
`src/kpnn2/_parse.py` or `src/kpnn2/_aggregation/`. Audit every
message built in them, including message parts built by
helpers and constants in those modules. If the user names no
modules, ask. Do not audit the whole package in one session:
quality drops over 150 messages.

## Read, in this order

1. `AGENTS.md`.
2. `dev/src_guidelines.txt`, section 3. Its message format and
   field rules are the standard you audit against.
3. `CONTEXT.md` **Errors** and **Docs terminology**, plus the
   sections on the behavior of the modules in scope. It is
   long: search it, do not read it whole.
4. The modules in scope, in full.
5. The version and the state of the tree:

   ```bash
   .venv/bin/python -c "import kpnn2; print(kpnn2.__version__)"
   git rev-parse --short HEAD
   git status --short
   ```

## What a message is

Every call in scope to `caller_error(...)`,
`warn_at_caller(...)`, or `internal_error(...)` from
`src/kpnn2/_errors.py`. A helper that builds a message for
several call sites counts once, in the module that defines it;
note its call sites.

## Render, then judge

Judge what the caller sees, not the source string. For each
message:

1. Trigger it through the public API with a small input, for
   example an edgelist of a few nodes, and print `str(error)`
   or the warning text. Use `.venv/bin/python`. Keep the
   snippets in the report directory as `render.py`.
2. If no public call reaches it in a few lines, render it by
   hand from the source with realistic values and mark it
   `rendered by hand`. Internal errors are always rendered by
   hand: they need a kpnn2 bug.
3. If the message has a `Fix`, apply the fix to your input and
   run again. The fix works if that error is gone. Record a fix
   that does not work.

## Checks

**Headline**

- States what is wrong in the caller's terms: argument names,
  node names, public names. No private names or internal
  layout terms.
- Matches the condition the code checks: not narrower, not
  broader.
- Makes sense alone, as the only line a log or test summary
  shows.
- With several problems, sums all of them up.

**Got**

- Present when a received value or an offending part of the
  input exists and showing it helps.
- Shows the offending part, not the whole input. A value is
  built with `_validate.describe`. No tensor values.
- A list goes through `listed`: at most ten items, then
  ", and N more".

**Expected**

- Present only when the headline does not say it already.
  Never repeats the headline.
- For a large set of valid names, says where to find them, for
  example `spec.nodes`.

**Did you mean**

- Present on every lookup of a name in a set of valid names,
  and nowhere else.
- The suggestions are plausible for the input you rendered.

**Fix**

- Present when the fix is not obvious from the lines above it.
  Absent when it would only restate them.
- Works: see step 3 of **Render, then judge**.
- Uses only the public API: names in `kpnn2.__all__` and their
  documented arguments. No private modules.
- Does not contradict `CONTEXT.md`, for example by suggesting a
  workaround a locked decision rules out.
- When almost no names match, says to check the kind of
  identifier (gene symbols vs Ensembl IDs, for example).
- For a warning, says what to do so the warning stops.

**Whole message**

- `Got`, `Expected`, and `Did you mean` written by kpnn2 end
  without a period.
- Several problems: the items in a field are separated by "; ".
- Nothing a field should carry is buried in the headline, and
  no field carries another field's content.

**Across messages**

- The same rule checked in two places gives the same message.
  The two parsers share their edgelist checks, for example.
- The same thing is named the same way everywhere, in the
  terms of `CONTEXT.md` **Docs terminology**.

## Rules

- **Report, do not edit.** Do not edit `src/`, `tests/`,
  `docs/`, `CONTEXT.md`, `AGENTS.md`, or `dev/`. The maintainer
  accepts findings, and they are implemented in a separate
  change.
- **Do not report the format.** Labels, order, line breaks,
  the headline's and `Fix`'s final period, and empty fields are
  enforced by tests.
- **Report only what you confirmed.** Every finding cites
  `file:line` and the rendered message.
- **No rewording for taste.** Report a message only when
  accuracy, usefulness, field choice, or consistency is at
  stake, and name which.
- **Messages only.** If the check itself looks wrong, for
  example it rejects valid input, list it under **Caveats**.
  Do not propose a new message for it.
- **Write only** the report directory
  `out/error_audit/<YYYY-MM-DD>_<short-sha>/`. Add `-dirty` to
  the name if `git status --short` was not empty. Name the
  report after the scope, for example `_parse.md` or
  `_aggregation.md`. Everything under `out/` is gitignored, so
  the session ends without a commit message.

## Report

````markdown
# Error-message audit: <scope>

kpnn2 <version>, <short sha>, tree <clean | dirty>.
Modules: <list>.

## Summary

<N> messages checked: <N> ok, <N> to change.
Findings: accuracy <N>, usefulness <N>, field choice <N>,
consistency <N>.

## Findings

### F1. <file>:<line>, <kind>

Current:

```
<rendered message>
```

Problem: <what a caller would get wrong or miss>

Proposed:

```
<rendered message>
```

Breaks: <tests whose `match=` or expected text holds the old
wording, and docs pages or notebooks that quote it; or none>

## All messages

| # | Location | Reached from | Type | Verdict | Finding |
|---|---|---|---|---|---|
| 1 | `_parse.py:384` | `parse_layered` | error | ok | |
| 2 | `_parse.py:541` | `parse_layered` | error | change | F1 |

## Caveats

<messages rendered by hand, fixes you could not run, checks
that look wrong, anything unclear>
````

Find what a change breaks with a search of `tests/` and
`docs/` for a distinctive part of the old text.

Reply with the report path, the counts from **Summary**, and
the three findings that matter most.
