# MiMo-V2-Flash

## Overview

MiMo-V2-Flash is a state-of-the-art Mixture of Experts (MoE) language model that leverages FlashAttention for efficient inference. It builds upon the successful MiMo-V2 architecture with optimized attention mechanisms for improved performance and lower memory consumption.

## Key Features

- **MoE Architecture**: Utilizes a mixture of experts to achieve high-quality results with efficient compute utilization
- **FlashAttention**: Implements FlashAttention for faster and more memory-efficient attention computations
- **High Scalability**: Designed to scale efficiently across multiple devices
- **Advanced Routing**: Sophisticated expert routing mechanisms to ensure optimal expert utilization
- **Wide Vocabulary Support**: Large vocabulary size for comprehensive language understanding

## Model Configuration

The MiMo-V2-Flash model is configured with the following key parameters:

| Parameter | Description |
|-----------|-------------|
| vocab_size | The size of the vocabulary used by the model |
| hidden_size | The dimensionality of the hidden layers |
| intermediate_size | The dimensionality of the intermediate layers in the feed-forward network |
| moe_intermediate_size | The dimensionality of the intermediate layers in the MoE experts |
| num_hidden_layers | The number of transformer layers |
| num_attention_heads | The number of attention heads |
| num_key_value_heads | The number of key-value heads (for grouped query attention) |
| num_experts | The total number of experts in the MoE layer |
| num_experts_per_tok | The number of experts to route each token to |
| rms_norm_eps | The epsilon value for RMS normalization |
| rope_theta | The theta value for rotary positional embeddings |
| max_position_embeddings | The maximum sequence length supported by the model |

## Usage Examples

### Running MiMo-V2-Flash with the CLI

```bash
# Run with default settings
./mistralrs-server -i run -m path/to/mimov2-flash-model

# Run with ISQ quantization for better performance
./mistralrs-server -i --isq 8 run -m path/to/mimov2-flash-model

# Run with specific architecture specified
./mistralrs-server -i run -m path/to/mimov2-flash-model --architecture mimov2flash
```

### Python API Usage

```python
import mistralrs

# Initialize the model
model = mistralrs.Runner(
    which=mistralrs.Which.Plain(
        model_id="path/to/mimov2-flash-model",
        architecture="mimov2flash"
    )
)

# Generate text
result = model.chat_completion(
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(result.choices[0].message.content)
```

## Performance Optimization

### Quantization

MiMo-V2-Flash supports In-place Quantization (ISQ) for improved performance and reduced memory usage:

```bash
./mistralrs-server -i --isq 8 run -m path/to/mimov2-flash-model
```

### Device Mapping

For large models, you can use device mapping to distribute the model across multiple GPUs:

```bash
./mistralrs-server -i --device-map auto run -m path/to/mimov2-flash-model
```

### Per-Layer Topology

Fine-tune quantization settings per layer for optimal performance:

```bash
./mistralrs-server -i --topology topologies/isq.yml run -m path/to/mimov2-flash-model
```

Example topology file (`topologies/isq.yml`):

```yaml
# Early layers: lower quantization for embeddings
0-8:
  isq: Q3K
# Middle layers: balanced quantization
8-24:
  isq: Q4K
# Final layers: higher quality for output
24-32:
  isq: Q6K
```

## Limitations

- Currently, MiMo-V2-Flash does not support LoRA or X-LoRA adapters
- Requires compatible hardware with FlashAttention support for optimal performance
- May require significant memory resources for full-precision inference

## Troubleshooting

### Model Loading Issues

If you encounter issues loading the model, ensure:

1. You're using the correct architecture name: `mimov2flash`
2. The model files are complete and properly formatted
3. You have sufficient memory available

### Performance Issues

For better performance:

1. Use ISQ quantization with appropriate bit width
2. Ensure FlashAttention is enabled (available on compatible NVIDIA GPUs)
3. Consider using device mapping for large models
4. Adjust the number of experts per token based on your hardware capabilities

## Support

For issues, questions, or contributions, please visit the [mistral.rs GitHub repository](https://github.com/EricLBuehler/mistral.rs) or join our [Discord community](https://discord.gg/SZrecqK8qw).
