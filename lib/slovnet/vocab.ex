defmodule Slovnet.Vocab do
  @moduledoc "Vocabulary mapping between tokens and integer IDs."

  defstruct [:items, :item_ids, :pad_id]

  @type t :: %__MODULE__{
          items: [String.t()],
          item_ids: %{String.t() => non_neg_integer()},
          pad_id: non_neg_integer()
        }

  @pad "<pad>"
  @unk "<unk>"

  def new(items) do
    item_ids = items |> Enum.with_index() |> Map.new()
    pad_id = Map.get(item_ids, @pad, 0)

    %__MODULE__{
      items: items,
      item_ids: item_ids,
      pad_id: pad_id
    }
  end

  def encode(%__MODULE__{} = vocab, item) do
    Map.get(vocab.item_ids, item, unk_id(vocab))
  end

  def decode(%__MODULE__{items: items}, id) when is_integer(id) do
    Enum.at(items, id, @unk)
  end

  defp unk_id(%__MODULE__{item_ids: item_ids}) do
    Map.get(item_ids, @unk, 0)
  end
end
