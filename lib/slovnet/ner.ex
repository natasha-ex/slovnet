defmodule Slovnet.NER do
  @moduledoc """
  Named Entity Recognition for Russian text.

  Recognizes PER (persons), LOC (locations), and ORG (organizations).

  ## Usage

      ner = Slovnet.NER.load()
      spans = Slovnet.NER.extract(ner, "Владимир Путин встретился с Ангелой Меркель в Кремле.")
      # [%{type: "PER", text: "Владимир Путин"}, %{type: "PER", text: "Ангелой Меркель"}, %{type: "LOC", text: "Кремле"}]

      # Batch mode — single forward pass for multiple texts:
      results = Slovnet.NER.extract_batch(ner, ["Путин в Кремле.", "Сегодня погода."])
      # [[%{type: "PER", ...}, %{type: "LOC", ...}], []]
  """

  defstruct [:model, :words_vocab, :shapes_vocab, :tags_vocab]

  @type t :: %__MODULE__{}
  @type span :: %{
          type: String.t(),
          text: String.t(),
          start: non_neg_integer(),
          stop: non_neg_integer()
        }

  alias Slovnet.{BIO, Model, Navec, Pack, Shape, Tokenizer, Vocab}

  @spec load(keyword()) :: t()
  def load(opts \\ []) do
    models_dir = Keyword.get(opts, :models_dir, default_models_dir())

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
      tags_vocab: Vocab.new(tags_items)
    }
  end

  @spec extract(t(), String.t()) :: [span()]
  def extract(%__MODULE__{} = ner, text) when is_binary(text) do
    tokens = Tokenizer.tokenize(text)

    {word_ids, shape_ids} = encode(ner, [Enum.map(tokens, & &1.text)])
    pad_mask = Nx.equal(word_ids, ner.words_vocab.pad_id)

    emissions = Model.run(ner.model, word_ids, shape_ids, pad_mask)
    [tag_ids] = Model.decode_crf(ner.model, emissions, pad_mask)

    tags = Enum.map(tag_ids, &Vocab.decode(ner.tags_vocab, &1))
    build_spans(text, tokens, tags)
  end

  @spec extract_batch(t(), [String.t()]) :: [[span()]]
  def extract_batch(%__MODULE__{}, []), do: []

  def extract_batch(%__MODULE__{} = ner, texts) when is_list(texts) do
    token_lists = Enum.map(texts, &Tokenizer.tokenize/1)
    word_lists = Enum.map(token_lists, fn tokens -> Enum.map(tokens, & &1.text) end)

    {word_ids, shape_ids} = encode(ner, word_lists)
    pad_mask = Nx.equal(word_ids, ner.words_vocab.pad_id)

    emissions = Model.run(ner.model, word_ids, shape_ids, pad_mask)
    all_tag_ids = Model.decode_crf(ner.model, emissions, pad_mask)

    [texts, token_lists, all_tag_ids]
    |> Enum.zip()
    |> Enum.map(fn {text, tokens, tag_ids} ->
      tags =
        tag_ids
        |> Enum.take(length(tokens))
        |> Enum.map(&Vocab.decode(ner.tags_vocab, &1))

      build_spans(text, tokens, tags)
    end)
  end

  defp build_spans(text, tokens, tags) do
    tokens
    |> BIO.spans_from_bio(tags)
    |> Enum.map(fn span ->
      %{
        type: span.type,
        text: String.slice(text, span.start, span.stop - span.start),
        start: span.start,
        stop: span.stop
      }
    end)
  end

  defp encode(%__MODULE__{} = ner, word_lists) do
    max_len = word_lists |> Enum.map(&length/1) |> Enum.max(fn -> 0 end)

    {word_rows, shape_rows} =
      word_lists
      |> Enum.map(fn words ->
        word_ids = Enum.map(words, &Vocab.encode(ner.words_vocab, String.downcase(&1)))

        shape_ids =
          Enum.map(words, fn w ->
            Vocab.encode(ner.shapes_vocab, Shape.word_shape(w))
          end)

        {pad(word_ids, max_len, ner.words_vocab.pad_id),
         pad(shape_ids, max_len, ner.shapes_vocab.pad_id)}
      end)
      |> Enum.unzip()

    {Nx.tensor(word_rows, type: :s64), Nx.tensor(shape_rows, type: :s64)}
  end

  defp pad(list, max_len, pad_id) do
    list ++ List.duplicate(pad_id, max_len - length(list))
  end

  defp default_models_dir do
    :slovnet
    |> :code.priv_dir()
    |> List.to_string()
    |> Path.join("models")
  end
end
