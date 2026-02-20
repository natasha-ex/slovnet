defmodule Slovnet.BIO do
  @moduledoc "BIO tag parsing and span extraction."

  defstruct [:type, :start, :stop]

  @type t :: %__MODULE__{type: String.t(), start: non_neg_integer(), stop: non_neg_integer()}

  def spans_from_bio(tokens, tags) do
    {spans, current} =
      Enum.zip(tokens, tags)
      |> Enum.reduce({[], nil}, fn {token, tag}, {spans, current} ->
        case parse_bio(tag) do
          {"B", type} ->
            spans = if current, do: [current | spans], else: spans
            {spans, %__MODULE__{type: type, start: token.start, stop: token.stop}}

          {"I", type} ->
            if current && current.type == type do
              {spans, %{current | stop: token.stop}}
            else
              spans = if current, do: [current | spans], else: spans
              {spans, %__MODULE__{type: type, start: token.start, stop: token.stop}}
            end

          _ ->
            spans = if current, do: [current | spans], else: spans
            {spans, nil}
        end
      end)

    spans = if current, do: [current | spans], else: spans
    Enum.reverse(spans)
  end

  defp parse_bio("O"), do: {"O", nil}
  defp parse_bio("B-" <> type), do: {"B", type}
  defp parse_bio("I-" <> type), do: {"I", type}
  defp parse_bio(_), do: {"O", nil}
end
