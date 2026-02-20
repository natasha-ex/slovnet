defmodule Slovnet.Model do
  @moduledoc """
  Neural NER model: Navec embedding + shape embedding → CNN encoder → CRF head.

  Uses `defn` for JIT-compiled inference — the entire forward pass compiles
  into a single XLA computation.
  """

  import Nx.Defn

  defstruct [:params, :navec_indexes, :navec_codes, :predict_fn]

  alias Slovnet.Pack

  def load(ner_path, navec) do
    {:ok, model_json} = Pack.read_json(ner_path, "model.json")
    arrays = load_arrays(ner_path, model_json)
    layers = load_encoder_layers(model_json["encoder"]["layers"], arrays)

    head = model_json["head"]

    params = %{
      shape_weight: arrays[model_json["emb"]["shape"]["weight"]["array"]],
      proj_weight: arrays[head["proj"]["weight"]["array"]],
      proj_bias: arrays[head["proj"]["bias"]["array"]],
      crf_transitions: arrays[head["crf"]["transitions"]["array"]],
      layers: layers
    }

    predict_fn = Nx.Defn.jit(&predict/5, compiler: EXLA)

    %__MODULE__{
      params: params,
      navec_indexes: navec.indexes,
      navec_codes: navec.codes,
      predict_fn: predict_fn
    }
  end

  def run(%__MODULE__{} = model, word_ids, shape_ids, pad_mask) do
    idx = navec_word_indices(model.navec_indexes, word_ids)
    model.predict_fn.(model.params, model.navec_codes, idx, shape_ids, pad_mask)
  end

  defp navec_word_indices(indexes, word_ids) do
    flat = Nx.reshape(word_ids, {:auto})
    Nx.take(indexes, flat) |> Nx.as_type(:s64)
  end

  @doc false
  defn predict(params, navec_codes, navec_idx, shape_ids, pad_mask) do
    word_emb = navec_gather(navec_codes, navec_idx)
    word_emb = reshape_emb(word_emb, shape_ids)

    shape_emb = Nx.take(params.shape_weight, shape_ids)
    x = Nx.concatenate([word_emb, shape_emb], axis: -1)

    x = encode(x, params.layers, pad_mask)
    linear(x, params.proj_weight, params.proj_bias)
  end

  deftransformp reshape_emb(word_emb, shape_ids) do
    emb_dim = elem(Nx.shape(word_emb), 1)
    out_shape = shape_ids |> Nx.shape() |> Tuple.to_list() |> Kernel.++([emb_dim]) |> List.to_tuple()
    Nx.reshape(word_emb, out_shape)
  end

  defnp encode(x, layers, pad_mask) do
    x = Nx.transpose(x, axes: [0, 2, 1])
    mask = Nx.new_axis(pad_mask, 1)

    x = cnn_layer(x, mask, layers[0])
    x = cnn_layer(x, mask, layers[1])
    x = cnn_layer(x, mask, layers[2])

    Nx.transpose(x, axes: [0, 2, 1])
  end

  defnp cnn_layer(x, mask, layer) do
    x = Nx.conv(x, layer.weight, padding: [{1, 1}])
    x = Nx.add(x, Nx.reshape(layer.bias, {1, :auto, 1}))
    x = Nx.max(x, 0)

    mean = Nx.reshape(layer.bn_mean, {1, :auto, 1})
    std = Nx.reshape(layer.bn_std, {1, :auto, 1})
    w = Nx.reshape(layer.bn_weight, {1, :auto, 1})
    b = Nx.reshape(layer.bn_bias, {1, :auto, 1})

    x = (x - mean) / std * w + b

    expanded_mask = Nx.broadcast(mask, Nx.shape(x))
    Nx.select(expanded_mask, Nx.tensor(0.0, type: Nx.type(x)), x)
  end

  defnp linear(input, weight, bias) do
    Nx.dot(input, [2], weight, [0]) + bias
  end

  defnp navec_gather(codes, idx) do
    {n, qdim} = Nx.shape(idx)
    {_qdim, _centroids, chunk} = Nx.shape(codes)

    q_range = Nx.iota({1, qdim}, type: :s64) |> Nx.broadcast({n, qdim})
    indices = Nx.stack([q_range, idx], axis: -1)

    gathered = Nx.gather(codes, indices)
    Nx.reshape(gathered, {n, qdim * chunk})
  end

  def decode_crf(%__MODULE__{params: params}, emissions, pad_mask) do
    {batch_size, seq_len, tags_num} = Nx.shape(emissions)

    emissions_list = Nx.to_list(emissions)
    mask_list = Nx.to_list(pad_mask)
    trans = Nx.to_list(params.crf_transitions)

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
    layers_json
    |> Enum.with_index()
    |> Map.new(fn {layer, i} ->
      conv = layer["conv"]
      norm = layer["norm"]

      {i,
       %{
         weight: arrays[conv["weight"]["array"]],
         bias: arrays[conv["bias"]["array"]],
         bn_weight: arrays[norm["weight"]["array"]],
         bn_bias: arrays[norm["bias"]["array"]],
         bn_mean: arrays[norm["mean"]["array"]],
         bn_std: arrays[norm["std"]["array"]]
       }}
    end)
  end
end
