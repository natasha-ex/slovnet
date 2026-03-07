defmodule Slovnet.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :slovnet,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      description: "Russian NER and morphological tagging (port of Natasha/Slovnet)",
      package: package()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.7"},
      {:exla, "~> 0.7", optional: true},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_dna, "~> 1.0", only: [:dev, :test], runtime: false},
      {:ex_slop, "~> 0.1", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      ci: [
        "compile --warnings-as-errors",
        "cmd MIX_ENV=test mix test",
        "credo --strict --min-priority high",
        "dialyzer",
        "ex_dna"
      ]
    ]
  end

  defp package do
    [
      maintainers: ["Danila Poyarkov"],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/natasha-ex/slovnet"},
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE)
    ]
  end
end
