# Dataset locations

`generated/` contains the current 500 graph corpus produced by
`generator.DatasetGenerator`, already split by graph into `train`, `val`, and
`test` before any Python preprocessing.

`raw_graphs/` is reserved for future exports from a real compiler frontend.
`processed/` is reserved for optional cached PyG `.pt` artifacts; the current
loader intentionally reads JSON so the source schema remains inspectable.
