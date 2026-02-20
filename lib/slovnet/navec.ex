defmodule Slovnet.Navec do
  @moduledoc """
  Navec word embeddings using product quantization.

  300-dimensional embeddings stored as 100 sub-vectors × 256 centroids × 3 dims.
  Each word maps to 100 uint8 centroid indices (~25MB for 250K words).
  """

  defstruct [:id, :vocab, :indexes, :codes, :codes_list]

  @type t :: %__MODULE__{
          id: String.t(),
          vocab: Slovnet.Vocab.t(),
          indexes: Nx.Tensor.t(),
          codes: Nx.Tensor.t(),
          codes_list: list()
        }

  alias Slovnet.Vocab

  def load(path) do
    {:ok, meta} = Slovnet.Pack.read_json(path, "meta.json")
    {:ok, vocab_bin} = Slovnet.Pack.read_tar(path, "vocab.bin")
    {:ok, pq_bin} = Slovnet.Pack.read_tar(path, "pq.bin")

    vocab = load_vocab(vocab_bin)
    {indexes, codes} = load_pq(pq_bin)

    # Precompute codes as tuples for fast scalar lookup
    codes_tuples =
      codes
      |> Nx.to_list()
      |> Enum.map(fn sub ->
        sub |> Enum.map(&List.to_tuple/1) |> List.to_tuple()
      end)
      |> List.to_tuple()

    %__MODULE__{
      id: meta["id"],
      vocab: vocab,
      indexes: indexes,
      codes: codes,
      codes_list: codes_tuples
    }
  end

  def lookup(%__MODULE__{} = navec, word_ids) when is_list(word_ids) do
    ids_tensor = Nx.tensor(word_ids, type: :s64)
    lookup_tensor(navec, ids_tensor)
  end

  def lookup_tensor(%__MODULE__{indexes: indexes, codes_list: codes_tuples}, word_ids) do
    flat = Nx.reshape(word_ids, {:auto})
    idx = Nx.take(indexes, flat)
    {_n, qdim} = Nx.shape(idx)
    chunk = tuple_size(elem(elem(codes_tuples, 0), 0))

    idx_list = Nx.to_list(idx)

    result =
      for word_idx <- idx_list do
        for q <- 0..(qdim - 1), reduce: [] do
          acc ->
            centroid_id = Enum.at(word_idx, q)
            sub = elem(codes_tuples, q)
            values = elem(sub, centroid_id)
            acc ++ Tuple.to_list(values)
        end
      end

    tensor = Nx.tensor(result, type: :f32)
    original_shape = Nx.shape(word_ids) |> Tuple.to_list()
    Nx.reshape(tensor, List.to_tuple(original_shape ++ [qdim * chunk]))
  end

  defp load_vocab(bin) do
    decompressed = :zlib.gunzip(bin)

    <<size::little-unsigned-32, rest::binary>> = decompressed
    counts_size = size * 4
    <<_counts::binary-size(counts_size), text::binary>> = rest

    words = String.split(text, "\n", trim: false)
    Vocab.new(words)
  end

  defp load_pq(bin) do
    <<vectors::little-unsigned-32, _dim::little-unsigned-32, qdim::little-unsigned-32,
      centroids::little-unsigned-32, rest::binary>> = bin

    indexes_size = vectors * qdim
    <<indexes_bin::binary-size(indexes_size), codes_bin::binary>> = rest

    indexes = Nx.from_binary(indexes_bin, :u8) |> Nx.reshape({vectors, qdim})
    chunk = div(byte_size(codes_bin), qdim * centroids * 4)
    codes = Nx.from_binary(codes_bin, :f32) |> Nx.reshape({qdim, centroids, chunk})

    {indexes, codes}
  end
end
