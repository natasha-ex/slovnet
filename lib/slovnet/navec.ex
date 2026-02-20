defmodule Slovnet.Navec do
  @moduledoc """
  Navec word embeddings using product quantization.

  300-dimensional embeddings stored as 100 sub-vectors × 256 centroids × 3 dims.
  Each word maps to 100 uint8 centroid indices (~25MB for 250K words).
  """

  defstruct [:id, :vocab, :indexes, :codes]

  @type t :: %__MODULE__{
          id: String.t(),
          vocab: Slovnet.Vocab.t(),
          indexes: Nx.Tensor.t(),
          codes: Nx.Tensor.t()
        }

  alias Slovnet.Vocab

  def load(path) do
    {:ok, meta} = Slovnet.Pack.read_json(path, "meta.json")
    {:ok, vocab_bin} = Slovnet.Pack.read_tar(path, "vocab.bin")
    {:ok, pq_bin} = Slovnet.Pack.read_tar(path, "pq.bin")

    vocab = load_vocab(vocab_bin)
    {indexes, codes} = load_pq(pq_bin)

    %__MODULE__{
      id: meta["id"],
      vocab: vocab,
      indexes: indexes,
      codes: codes
    }
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
