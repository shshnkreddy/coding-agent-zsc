# Instructions for independently generated policies that follow H-group Level 1 conventions

These are experiment instructions. The [copied documents](INDEX.md) supply the
conventions themselves; read them before designing the policy.

## Shared conventions, independent implementations

Your partners are independently generated policies that follow H-group Level 1
conventions. Every partner receives this same pinned bundle. Use its documented meanings as the shared
coordination protocol. Seek competent play and robust coordination with other
convention-following agents. Assume shared convention meanings, while allowing
differences in aspects such as implementation, tactical choices, risk thresholds,
and tie-breaking wherever the conventions leave choices open. Do not replace this protocol with a
private signaling scheme, or require your partner to know your source code,
optimizer, memory representation, action ranking, risk threshold, or random seed.
Do not import higher-level conventions from prior knowledge or external sources.

Independent implementations may select different convention-consistent clues,
plays or discards, and make different probabilistic judgments where the
documentation leaves uncertainty. Implementation mistakes are also possible;
an observed mistake does not imply a different agreed convention system.
Infer what an action communicates through Level 1 and public history. Do not
infer a private message merely because your own policy would have selected a
different action. Avoid relying on undisclosed tie-breaks between otherwise valid
choices. Positional choices explicitly specified by H-group are shared conventions,
and must use the runtime mapping.

Optimize the objective in the task specification. For XP, seek strong cross-play
with independently generated policies that follow H-group Level 1 conventions
in both seats. Self-play can diagnose
competence and convention mistakes, but does not measure that cross-play target.
For SP, maximize self-play return subject to the same Level 1 requirements; the
SP runner's score-based selection remains unchanged.

## Interpret and give clues using the copied rules

- Track chop, newly touched versus previously clued cards, and clue focus using
  the pre-clue hand. Read [single-card focus](docs/beginner/single-card-focus.md)
  and the [Level 1 refinements](docs/level-1.md).
- Distinguish play clues, delayed play clues, and the permitted chop saves.
  Saving a card is not a promise that it can be played immediately. A clue
  focusing a non-chop card is a play clue; a chop clue may remain ambiguous.
  Preserve all documented play/save interpretations when neither is ruled out.
- Apply Good Touch, Save, and Minimum Clue Value principles, including the
  documented exceptions and the early-game requirement. Do not treat every
  touched card as immediately playable or every rank-2/rank-5 clue as a save.
- Apply the rank-2 visible-copy rule from the clue giver's information, including
  the reference's double-chop exception where its premises are knowable. Never
  inspect your hidden hand to decide whether a copy is visible to you.
- Read prompts (including self-prompts), finesse position, prompt precedence,
  and the treatment of already finessed cards. Use only connections supported by
  the documented evidence and turn structure. The runtime guide explains why
  ordinary three-player finesse examples do not become two-player self-finesses.
- Keep the literal legal-action mask authoritative. The guide sometimes calls a
  convention-breaking clue "illegal"; this is different from an invalid game
  action. A legal forced clue does not authorize a higher-level stall signal.

## Knowledge, memory, and uncertainty

Keep literal positive and negative hint constraints distinct from convention
inferences. The native knowledge features encode literal information; they do not
automatically implement H-group or your partner's intent. Add observation-safe
inference and fixed-shape private memory as needed.

Treat card notes as private memory about your own and your partner's publicly
established information. Carry notes with cards when slots shift, clear them for
new cards, and ignore empty slots. Maintain relevant clue history and dependencies
across turns instead of recomputing old clues as though they were given now.

When evidence is ambiguous, preserve the alternatives. When public evidence
contradicts an inferred promise, revise the inference; never override literal
facts or invent a higher-level convention to explain the contradiction. Choose
subsequent actions whose meanings remain understandable under Level 1. This
does not require one particular belief architecture or numerical risk threshold.

## Development review

In your normal strategy notes and ledger, explain which documented convention
an intentional signal uses, what the recipient can infer, and which uncertainty
remains. Inspect failures involving focus, save/play ambiguity, positional
mapping, delayed dependencies, or assumptions about a partner's action choice.
Use only the task's permitted development tools and available evidence.

No convention implementation, partner policy, hidden state, or shared private
memory is supplied by this bundle. Implement a policy within the existing API
and information boundary.
