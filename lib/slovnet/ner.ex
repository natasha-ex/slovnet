defmodule Slovnet.NER do
  @moduledoc """
  Named Entity Recognition for Russian text.

  Recognizes PER (persons), LOC (locations), and ORG (organizations).

  ## Usage

      ner = Slovnet.NER.load()
      spans = Slovnet.NER.extract(ner, "Владимир Путин встретился с Ангелой Меркель в Кремле.")
      # [%{type: "PER", text: "Владимир Путин"}, %{type: "PER", text: "Ангелой Меркель"}, %{type: "LOC", text: "Кремле"}]
  """

  defstruct [:model, :words_vocab, :shapes_vocab, :tags_vocab, :batch_size]

  alias Slovnet.{BIO, Model, Navec, Pack, Shape, Tokenizer, Vocab}

  @default_batch_size 8

  def load(opts \\ []) do
    models_dir = Keyword.get(opts, :models_dir, default_models_dir())
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)

    navec_path = Path.join(models_dir, "navec_news_v1_1B_250K_300d_100q.tar")
    ner_path = Path.join(models_dir, "slovnet_ner_news_v1.tar")

    navec = Navec.load(navec_path)
    model = Model.load(ner_path, navec)

    {:ok, words_items} = Pack.read_vocab(ner_path, "vocabs/word.gz")
    {:ok, shapes_items} = Pack.read_vocab(ner_path, "vocabs/shape.gz")
    {:ok, tags_items} = Pack.read_vocab(ner_path, "vocabs/tag.gz")

    %__MODULE__{
      model: model,
      words_vocab: Vocab.new(words_items),
      shapes_vocab: Vocab.new(shapes_items),
      tags_vocab: Vocab.new(tags_items),
      batch_size: batch_size
    }
  end

  def extract(%__MODULE__{} = ner, text) when is_binary(text) do
    tokens = Tokenizer.tokenize(text)
    words = Enum.map(tokens, & &1.text)

    {word_ids, shape_ids} = encode(ner, words)
    pad_mask = Nx.equal(word_ids, ner.words_vocab.pad_id)

    emissions = Model.forward(ner.model, word_ids, shape_ids, pad_mask)
    [tag_ids] = Model.decode_crf(ner.model, emissions, pad_mask)

    tags = Enum.map(tag_ids, &Vocab.decode(ner.tags_vocab, &1))
    spans = BIO.spans_from_bio(tokens, tags)

    Enum.map(spans, fn span ->
      %{
        type: span.type,
        text: String.slice(text, span.start, span.stop - span.start),
        start: span.start,
        stop: span.stop
      }
    end)
  end

  defp encode(%__MODULE__{} = ner, words) do
    word_ids =
      Enum.map(words, fn word ->
        Vocab.encode(ner.words_vocab, String.downcase(word))
      end)

    shape_ids =
      Enum.map(words, fn word ->
        shape = Shape.word_shape(word)
        Vocab.encode(ner.shapes_vocab, shape)
      end)

    {Nx.tensor([word_ids], type: :s64), Nx.tensor([shape_ids], type: :s64)}
  end

  defp default_models_dir do
    :slovnet
    |> :code.priv_dir()
    |> List.to_string()
    |> Path.join("models")
  end
end
