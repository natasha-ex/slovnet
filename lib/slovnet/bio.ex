defmodule Slovnet.BIO do
  @moduledoc "BIO tag parsing and span extraction."

  defstruct [:type, :start, :stop]

  @type t :: %__MODULE__{type: String.t(), start: non_neg_integer(), stop: non_neg_integer()}

  def spans_from_bio(tokens, tags) do
    {spans, current} =
      tokens
      |> Enum.zip(tags)
      |> Enum.reduce({[], nil}, &process_token/2)

    maybe_close(spans, current) |> Enum.reverse()
  end

  defp process_token({token, tag}, {spans, current}) do
    case parse_bio(tag) do
      {"B", type} ->
        {maybe_close(spans, current),
         %__MODULE__{type: type, start: token.start, stop: token.stop}}

      {"I", type} when current != nil and current.type == type ->
        {spans, %{current | stop: token.stop}}

      {"I", type} ->
        {maybe_close(spans, current),
         %__MODULE__{type: type, start: token.start, stop: token.stop}}

      _ ->
        {maybe_close(spans, current), nil}
    end
  end

  defp maybe_close(spans, nil), do: spans
  defp maybe_close(spans, current), do: [current | spans]

  defp parse_bio("O"), do: {"O", nil}
  defp parse_bio("B-" <> type), do: {"B", type}
  defp parse_bio("I-" <> type), do: {"I", type}
  defp parse_bio(_), do: {"O", nil}
end
