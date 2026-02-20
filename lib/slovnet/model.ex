defmodule Slovnet.Model do
  @moduledoc """
  Neural NER model: Navec embedding + shape embedding → CNN encoder → CRF head.

  All operations use Nx tensors for pure Elixir inference.
  """

  defstruct [:navec, :shape_weight, :encoder_layers, :proj_weight, :proj_bias, :crf_transitions]

  @type t :: %__MODULE__{
          navec: Slovnet.Navec.t(),
          shape_weight: Nx.Tensor.t(),
          encoder_layers: [map()],
          proj_weight: Nx.Tensor.t(),
          proj_bias: Nx.Tensor.t(),
          crf_transitions: Nx.Tensor.t()
        }

  alias Slovnet.{Navec, Pack}

  def load(ner_path, navec) do
    {:ok, model_json} = Pack.read_json(ner_path, "model.json")

    arrays = load_arrays(ner_path, model_json)
    encoder_layers = load_encoder_layers(model_json["encoder"]["layers"], arrays)

    head = model_json["head"]
    proj = head["proj"]
    crf = head["crf"]

    %__MODULE__{
      navec: navec,
      shape_weight: arrays[model_json["emb"]["shape"]["weight"]["array"]],
      encoder_layers: encoder_layers,
      proj_weight: arrays[proj["weight"]["array"]],
      proj_bias: arrays[proj["bias"]["array"]],
      crf_transitions: arrays[crf["transitions"]["array"]]
    }
  end

  def forward(%__MODULE__{} = model, word_ids, shape_ids, pad_mask) do
    word_emb = Navec.lookup_tensor(model.navec, word_ids)
    shape_emb = Nx.take(model.shape_weight, shape_ids)
    x = Nx.concatenate([word_emb, shape_emb], axis: -1)

    x = encode(x, model.encoder_layers, pad_mask)
    linear(x, model.proj_weight, model.proj_bias)
  end

  def decode_crf(%__MODULE__{crf_transitions: transitions}, emissions, pad_mask) do
    {batch_size, seq_len, tags_num} = Nx.shape(emissions)

    # Convert to lists for fast scalar access
    emissions_list = Nx.to_list(emissions)
    mask_list = Nx.to_list(pad_mask)
    trans = Nx.to_list(transitions)

    for b <- 0..(batch_size - 1) do
      b_emissions = Enum.at(emissions_list, b)
      b_mask = Enum.at(mask_list, b)

      score = Enum.at(b_emissions, 0)

      {score, history} =
        Enum.reduce(1..(seq_len - 1), {score, []}, fn t, {score, history} ->
          em = Enum.at(b_emissions, t)
          is_pad = Enum.at(b_mask, t) == 1

          if is_pad do
            {score, [List.duplicate(0, tags_num) | history]}
          else
            {new_score, indexes} =
              Enum.reduce(0..(tags_num - 1), {[], []}, fn j, {scores_acc, idx_acc} ->
                {best_score, best_idx} =
                  Enum.reduce(0..(tags_num - 1), {-1.0e30, 0}, fn i, {bs, bi} ->
                    v = Enum.at(score, i) + Enum.at(Enum.at(trans, i), j) + Enum.at(em, j)
                    if v > bs, do: {v, i}, else: {bs, bi}
                  end)

                {scores_acc ++ [best_score], idx_acc ++ [best_idx]}
              end)

            {new_score, [indexes | history]}
          end
        end)

      best = score |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)

      Enum.reduce(history, [best], fn indexes, [current | _] = tags ->
        [Enum.at(indexes, current) | tags]
      end)
    end
  end

  defp encode(x, layers, pad_mask) do
    x = Nx.transpose(x, axes: [0, 2, 1])
    mask = Nx.new_axis(pad_mask, 1)

    x =
      Enum.reduce(layers, x, fn layer, x ->
        x = conv1d(x, layer.weight, layer.bias, layer.padding)
        x = Nx.max(x, 0)
        x = batch_norm(x, layer.bn_weight, layer.bn_bias, layer.bn_mean, layer.bn_std)

        expanded_mask = Nx.broadcast(mask, Nx.shape(x))
        Nx.select(expanded_mask, Nx.tensor(0.0, type: Nx.type(x)), x)
      end)

    Nx.transpose(x, axes: [0, 2, 1])
  end

  defp conv1d(input, weight, bias, padding) do
    # input: {batch, channels, seq}, weight: {filters, channels, kernel}
    # Nx.conv expects input {batch, channels, spatial...}, kernel {filters, channels, spatial...}
    result = Nx.conv(input, weight, padding: [{padding, padding}])
    Nx.add(result, Nx.reshape(bias, {1, :auto, 1}))
  end

  defp batch_norm(input, weight, bias, mean, std) do
    mean = Nx.reshape(mean, {1, :auto, 1})
    std = Nx.reshape(std, {1, :auto, 1})
    weight = Nx.reshape(weight, {1, :auto, 1})
    bias = Nx.reshape(bias, {1, :auto, 1})

    input
    |> Nx.subtract(mean)
    |> Nx.divide(std)
    |> Nx.multiply(weight)
    |> Nx.add(bias)
  end

  defp linear(input, weight, bias) do
    {b, s, _d} = Nx.shape(input)
    {in_dim, out_dim} = Nx.shape(weight)
    flat = Nx.reshape(input, {b * s, in_dim})
    result = Nx.add(Nx.dot(flat, weight), bias)
    Nx.reshape(result, {b, s, out_dim})
  end

  defp load_arrays(ner_path, model_json) do
    weights = collect_weights(model_json, [])

    Map.new(weights, fn %{"shape" => shape, "dtype" => dtype, "array" => id} ->
      {:ok, tensor} = Pack.read_array(ner_path, "arrays/#{id}.bin", shape, dtype)
      {id, tensor}
    end)
  end

  defp collect_weights(map, acc) when is_map(map) do
    if Map.has_key?(map, "array") and Map.has_key?(map, "shape") do
      [map | acc]
    else
      Enum.reduce(map, acc, fn {_k, v}, acc -> collect_weights(v, acc) end)
    end
  end

  defp collect_weights(list, acc) when is_list(list) do
    Enum.reduce(list, acc, &collect_weights/2)
  end

  defp collect_weights(_, acc), do: acc

  defp load_encoder_layers(layers_json, arrays) do
    Enum.map(layers_json, fn layer ->
      conv = layer["conv"]
      norm = layer["norm"]

      %{
        weight: arrays[conv["weight"]["array"]],
        bias: arrays[conv["bias"]["array"]],
        padding: conv["padding"],
        bn_weight: arrays[norm["weight"]["array"]],
        bn_bias: arrays[norm["bias"]["array"]],
        bn_mean: arrays[norm["mean"]["array"]],
        bn_std: arrays[norm["std"]["array"]]
      }
    end)
  end
end
