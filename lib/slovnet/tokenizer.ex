defmodule Slovnet.Tokenizer do
  @moduledoc "Simple regex tokenizer for slovnet (matching Python's slovnet.token)."

  @token_re ~r/[а-яёa-z0-9]+(?:[-][а-яёa-z0-9]+)*|[^\s]/iu

  defstruct [:text, :start, :stop]

  @type t :: %__MODULE__{text: String.t(), start: non_neg_integer(), stop: non_neg_integer()}

  def tokenize(text) do
    byte_to_char = build_byte_to_char(text)

    Regex.scan(@token_re, text, return: :index)
    |> Enum.map(fn [{byte_start, byte_len}] ->
      token_text = binary_part(text, byte_start, byte_len)
      char_start = Map.fetch!(byte_to_char, byte_start)
      char_stop = Map.fetch!(byte_to_char, byte_start + byte_len)

      %__MODULE__{text: token_text, start: char_start, stop: char_stop}
    end)
  end

  defp build_byte_to_char(text) do
    text
    |> String.to_charlist()
    |> Enum.reduce({0, 0, %{0 => 0}}, fn codepoint, {byte_offset, char_idx, map} ->
      byte_len =
        cond do
          codepoint <= 0x7F -> 1
          codepoint <= 0x7FF -> 2
          codepoint <= 0xFFFF -> 3
          true -> 4
        end

      new_byte = byte_offset + byte_len
      new_char = char_idx + 1
      {new_byte, new_char, Map.put(map, new_byte, new_char)}
    end)
    |> elem(2)
  end
end
