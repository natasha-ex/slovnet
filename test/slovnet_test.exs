defmodule SlovnetTest do
  use ExUnit.Case

  setup_all do
    ner = Slovnet.NER.load()
    %{ner: ner}
  end

  describe "NER" do
    test "detects person names", %{ner: ner} do
      spans = Slovnet.NER.extract(ner, "Татьяна Голикова рассказала в среду.")
      assert [%{type: "PER", text: "Татьяна Голикова"}] = spans
    end

    test "detects multiple persons", %{ner: ner} do
      spans = Slovnet.NER.extract(ner, "Владимир Путин встретился с Ангелой Меркель в Кремле.")
      types = Enum.map(spans, & &1.type)
      texts = Enum.map(spans, & &1.text)

      assert "PER" in types
      assert "LOC" in types
      assert "Владимир Путин" in texts
      assert "Ангелой Меркель" in texts
      assert "Кремле" in texts
    end

    test "detects organizations", %{ner: ner} do
      spans = Slovnet.NER.extract(ner, "ООО «Газпром межрегионгаз» выступило истцом.")
      assert [%{type: "ORG"}] = spans
    end

    test "detects locations", %{ner: ner} do
      spans = Slovnet.NER.extract(ner, "Президент прибыл в Москву из Санкт-Петербурга.")

      locs =
        spans
        |> Enum.filter(&(&1.type == "LOC"))
        |> Enum.map(& &1.text)

      assert "Москву" in locs
      assert "Санкт-Петербурга" in locs
    end

    test "returns empty for text without entities", %{ner: ner} do
      spans = Slovnet.NER.extract(ner, "Сегодня хорошая погода.")
      assert spans == []
    end

    test "span positions are correct", %{ner: ner} do
      text = "Владимир Путин рассказал."
      [span | _] = Slovnet.NER.extract(ner, text)

      assert span.start == 0
      assert span.stop == 14
      assert String.slice(text, span.start, span.stop - span.start) == "Владимир Путин"
    end
  end

  describe "Shape" do
    test "Russian word shapes" do
      assert Slovnet.Shape.word_shape("Москва") == "RU_Xx"
      assert Slovnet.Shape.word_shape("москва") == "RU_xx"
      assert Slovnet.Shape.word_shape("МОСКВА") == "RU_XX"
      assert Slovnet.Shape.word_shape("в") == "RU_x"
      assert Slovnet.Shape.word_shape("В") == "RU_X"
    end

    test "English word shapes" do
      assert Slovnet.Shape.word_shape("Hello") == "EN_Xx"
      assert Slovnet.Shape.word_shape("hello") == "EN_xx"
    end

    test "number and punct shapes" do
      assert Slovnet.Shape.word_shape("123") == "NUM"
      assert Slovnet.Shape.word_shape(",") == "PUNCT_,"
      assert Slovnet.Shape.word_shape("...") == "PUNCT_OTHER"
    end
  end

  describe "Tokenizer" do
    test "tokenizes Russian text" do
      tokens = Slovnet.Tokenizer.tokenize("Привет мир!")
      texts = Enum.map(tokens, & &1.text)
      assert texts == ["Привет", "мир", "!"]
    end

    test "preserves character positions" do
      tokens = Slovnet.Tokenizer.tokenize("Москва — столица.")
      moscow = hd(tokens)
      assert moscow.start == 0
      assert moscow.stop == 6
    end
  end

  describe "Vocab" do
    test "encodes and decodes" do
      vocab = Slovnet.Vocab.new(["<pad>", "<unk>", "hello", "world"])
      assert Slovnet.Vocab.encode(vocab, "hello") == 2
      assert Slovnet.Vocab.encode(vocab, "missing") == 1
      assert Slovnet.Vocab.decode(vocab, 2) == "hello"
    end
  end
end
