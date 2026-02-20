defmodule Slovnet.Shape do
  @moduledoc "Computes word shape features for slovnet encoding."

  @ru_re ~r/^[а-яё]+$/iu
  @en_re ~r/^[a-z]+$/i
  @num_re ~r/^[+-]?\d+$/

  @puncts "!#$%&()[]\\/*+,.:;<=>?@^_{|}~-‐−‒⁃–—―`\"'«»„\u201c\u02bc\u02bb\u201d№…"
          |> String.graphemes()
          |> MapSet.new()

  def word_shape(word) do
    type = word_type(word)

    case type do
      t when t in ["RU", "EN"] -> "#{type}_#{word_outline(word)}"
      "PUNCT" -> punct_shape(word)
      _ -> type
    end
  end

  defp word_type(word) do
    cond do
      Regex.match?(@ru_re, word) -> "RU"
      Regex.match?(@en_re, word) -> "EN"
      Regex.match?(@num_re, word) -> "NUM"
      is_punct?(word) -> "PUNCT"
      true -> "OTHER"
    end
  end

  defp is_punct?(word) do
    word
    |> String.graphemes()
    |> Enum.all?(&MapSet.member?(@puncts, &1))
  end

  defp punct_shape(word) do
    if String.length(word) > 1 or not MapSet.member?(@puncts, word) do
      "PUNCT_OTHER"
    else
      "PUNCT_#{word}"
    end
  end

  defp word_outline(word) do
    len = String.length(word)

    cond do
      len == 1 and upper?(word) -> "X"
      len == 1 -> "x"
      upper?(word) -> "XX"
      lower?(word) -> "xx"
      title?(word) -> "Xx"
      dash_title?(word) -> "Xx-Xx"
      true -> "OTHER"
    end
  end

  defp upper?(word), do: word == String.upcase(word)
  defp lower?(word), do: word == String.downcase(word)

  defp title?(word) do
    String.length(word) > 1 and
      String.first(word) == String.upcase(String.first(word)) and
      String.slice(word, 1..-1//1) == String.downcase(String.slice(word, 1..-1//1))
  end

  defp dash_title?(word) do
    case String.split(word, "-", parts: 2) do
      [left, right] -> title?(left) and title?(right)
      _ -> false
    end
  end
end
