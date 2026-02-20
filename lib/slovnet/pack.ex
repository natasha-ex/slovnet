defmodule Slovnet.Pack do
  @moduledoc "Loads slovnet model packs (tar archives)."

  @doc "Reads a file from a tar archive."
  def read_tar(tar_path, member_name) do
    {:ok, files} = :erl_tar.extract(String.to_charlist(tar_path), [:memory])

    files
    |> Enum.find(fn {name, _data} -> List.to_string(name) == member_name end)
    |> case do
      nil -> {:error, :not_found}
      {_name, data} -> {:ok, data}
    end
  end

  @doc "Reads and parses a JSON file from a tar archive."
  def read_json(tar_path, member_name) do
    with {:ok, data} <- read_tar(tar_path, member_name) do
      {:ok, Jason.decode!(data)}
    end
  end

  @doc "Reads and decompresses a gzip vocab file from a tar archive."
  def read_vocab(tar_path, member_name) do
    with {:ok, data} <- read_tar(tar_path, member_name) do
      text = :zlib.gunzip(data)
      items = String.split(text, "\n", trim: false)
      {:ok, items}
    end
  end

  @doc "Reads a binary array from a tar archive."
  def read_array(tar_path, member_name, shape, dtype) do
    with {:ok, data} <- read_tar(tar_path, member_name) do
      tensor = binary_to_tensor(data, shape, dtype)
      {:ok, tensor}
    end
  end

  defp binary_to_tensor(data, shape, "float32") do
    Nx.from_binary(data, :f32) |> Nx.reshape(List.to_tuple(shape))
  end

  defp binary_to_tensor(data, shape, "uint8") do
    Nx.from_binary(data, :u8) |> Nx.reshape(List.to_tuple(shape))
  end

  defp binary_to_tensor(data, shape, "int64") do
    Nx.from_binary(data, :s64) |> Nx.reshape(List.to_tuple(shape))
  end
end
