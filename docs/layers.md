# Custom Layers

kmodels provides various custom layers like StochasticDepth, LayerScale, EfficientMultiheadSelfAttention, and more. These layers can be seamlessly integrated into your custom models and workflows.

## Usage

```python
import kmodels

# Example 1
layer = kmodels.layers.StochasticDepth(drop_path_rate=0.1)
output = layer(input_tensor, training=True)

# Example 2
window_partition = WindowPartition(window_size=7)
windowed_features = window_partition(features, height=28, width=28)
```
