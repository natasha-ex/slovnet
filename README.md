# Slovnet

Russian Named Entity Recognition (NER) for Elixir — detects person names (PER), locations (LOC), and organizations (ORG).

Pure Elixir port of [natasha/slovnet](https://github.com/natasha/slovnet). CNN+CRF architecture with Navec product-quantized embeddings. No Python, no ONNX — inference runs entirely on the BEAM via [Nx](https://github.com/elixir-nx/nx).

## Installation

Add the dependency to `mix.exs`:

```elixir
def deps do
  [
    {:slovnet, "~> 0.1"}
  ]
end
```

Download the model files (not included on Hex due to size):

```bash
mkdir -p priv/models

curl -L https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar \
  -o priv/models/navec_news_v1_1B_250K_300d_100q.tar

curl -L https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_ner_news_v1.tar \
  -o priv/models/slovnet_ner_news_v1.tar
```

### EXLA (recommended)

For ~100× speedup over the default `BinaryBackend`, add EXLA:

```elixir
def deps do
  [
    {:slovnet, "~> 0.1"},
    {:exla, "~> 0.7"}
  ]
end
```

```elixir
# config/config.exs
config :nx, default_backend: EXLA.Backend
```

## Usage

```elixir
ner = Slovnet.NER.load()

Slovnet.NER.extract(ner, "Владимир Путин встретился с Ангелой Меркель в Кремле.")
# [%{type: "PER", text: "Владимир Путин", start: 0, stop: 14},
#  %{type: "PER", text: "Ангелой Меркель", start: 28, stop: 43},
#  %{type: "LOC", text: "Кремле", start: 46, stop: 52}]
```

To load models from a custom directory:

```elixir
ner = Slovnet.NER.load(models_dir: "/path/to/models")
```

## Performance

~460μs per sentence (14 words) with EXLA on CPU.

## License

MIT — Danila Poyarkov
