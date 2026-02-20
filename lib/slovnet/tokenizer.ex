defmodule Slovnet.Tokenizer do
  @moduledoc "Simple regex tokenizer for slovnet (matching Python's slovnet.token)."

  @token_re ~r/[а-яёa-z0-9]+(?:[-][а-яёa-z0-9]+)*|[^\s]/iu

  defstruct [:text, :start, :stop]

  @type t :: %__MODULE__{text: String.t(), start: non_neg_integer(), stop: non_neg_integer()}

  def tokenize(text) do
    Regex.scan(@token_re, text, return: :index)
    |> Enum.map(fn [{byte_start, byte_len}] ->
      token_text = binary_part(text, byte_start, byte_len)
      char_start = text |> binary_part(0, byte_start) |> String.length()
      char_stop = char_start + String.length(token_text)

      %__MODULE__{text: token_text, start: char_start, stop: char_stop}
    end)
  end
end
