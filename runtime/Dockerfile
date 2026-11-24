# Build from the runtime/ directory.
FROM python:3.12-slim
RUN pip install --no-cache-dir 'jax[cpu]==0.4.33' 'flax==0.9.0' 'chex==0.1.87' 'numpy<2' 'omegaconf==2.3.0' 'gymnax==0.0.9' safetensors && pip install --no-cache-dir --no-deps 'jaxmarl==0.1.0'
COPY src/llm_policy /opt/zsc/llm_policy
COPY docker/llm_policy_jaxmarl /opt/zsc/jaxmarl
RUN python -c "import site,pathlib; (pathlib.Path(site.getsitepackages()[0])/'autoastra-zsc.pth').write_text(\"import sys; sys.path.insert(0, '/opt/zsc')\\n\")"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false CLAUDE_CODE_EFFORT_LEVEL=high
