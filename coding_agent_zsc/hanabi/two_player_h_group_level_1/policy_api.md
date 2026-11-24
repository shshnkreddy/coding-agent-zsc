# Policy interface

Keep POLICY_API_VERSION=1 and the starter's metadata. Implement init_agent() and agent_act(obs, reset, memory, rng).
Each initializer returns a private fixed-shape JAX PyTree. Each act call returns
(scalar integer action, next_memory, trace). Memory structure, shapes and
dtypes must match the value returned by the initializer. Honor reset; seats never share private memory.
Use only local observations and the supplied PRNG, never hidden state or reward.

Observations use HanabiObservation, imported from llm_policy.policy_api.
Trace is a dictionary whose string keys and contents you choose, mapping to
fixed-shape numeric or boolean JAX arrays. Return {} when no diagnostics are
needed. Keys, shapes and dtypes must remain fixed across calls for each act
function; different policies and seats may use different trace schemas.
There are no required diagnostic fields or predefined meanings. The runner
records actual actions separately. Trace is recorded only for inspection: it
is not passed into the next act call, shared with another seat, or used to
compute reward. Only next_memory is carried into the next decision, and that
memory is reset to the value returned by the initializer at episode boundaries.

See environment/README.md and environment/specification.json for observation
semantics, shapes, action encodings and rules. All implementations beyond the
bare interface are yours, including observation decoding and memory updates.
No supplied primitive libraries may be imported. Keep all policy code in final_solution.py. JAX, chex, dataclasses and fixed-shape Python utilities
remain available; runtime I/O, callbacks and dynamic Python control flow do not.


Use fitness-eval for official self-play scores.
