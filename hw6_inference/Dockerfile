FROM vllm/vllm-openai:v0.22.0-cu129@sha256:14f7be59bea024f60a7cb7076ce5666d7c72751354a3db22cb713af28cffa2cf

RUN python3 -m venv /opt/benchmark \
    && /opt/benchmark/bin/pip install --no-cache-dir aiperf==0.13.0 transformers==5.22.1
ENV PATH="/opt/benchmark/bin:${PATH}"

WORKDIR /homework
COPY benchmark.sh load_test.sh serve.sh ./
COPY data ./data
ENTRYPOINT []
CMD ["bash"]
