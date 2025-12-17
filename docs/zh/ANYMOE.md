# 从任何模型构建高效内存的 MoE 模型，只需几秒钟

AnyMoE 是一种动态高效创建 MoE 模型的技术。通过提供一组专家和小型预训练数据集，您可以在本地创建 MoE！

它具有以下特点：
- 将 AnyMoE 应用于任何支持的模型
    - `plain`
    - `vision-plain`
- 指定要应用 AnyMoE 的层以进行高效训练

论文：https://arxiv.org/abs/2405.19076

https://github.com/EricLBuehler/mistral.rs/assets/65165915/33593903-d907-4c08-a0ac-d349d7bf33de

> 注意：默认情况下，这可以创建 csv 损失图像。从源代码构建时（对于 Python 或 CLI），您可以使用 `--no-default-features` 命令行禁用此功能。如果网络不可用，这可能是必要的。

## 数据集
目前，AnyMoE 期望使用 JSON 数据集，其中包含一个顶级键 `row`，它是一个对象数组，包含键 `prompt`（字符串）、`expert`（整数）和 `image_urls`（可选的字符串数组）。例如：
```json
{
    "rows": [
        {
            "prompt": "Discuss the impact of Renaissance art on modern aesthetics",
            "expert": 0
        },
        {
            "prompt": "Explain the significance of the theory of relativity in modern physics",
            "expert": 1
        },
    ]
}  

```

对于视觉模型，`image_urls` 可以包含图像 URL/本地路径或 Base64 编码图像的数组。

## 专家
AnyMoE 专家可以是微调模型或 LoRA 适配器模型。只会从每个模型加载 mlp 层。专家必须是同构的：它们必须都是微调的或都是适配器。此外，可以指定应用 AnyMoE 的特定层。

> 注意：当使用 LoRA 适配器专家时，由于内存使用率较低，可能不需要设置应用 AnyMoE 的层。

### 带有微调专家的 TOML 选择器示例
```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
arch = "mistral"

[anymoe]
dataset_json = "examples/amoe.json"
prefix = "model.layers"
mlp = "mlp"
model_ids = ["HuggingFaceH4/zephyr-7b-beta"]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[anymoe.config]
hidden_size = 4096
expert_type = "fine_tuned"
```

### 带有 LoRA 适配器专家的 TOML 选择器示例
```toml
[model]
model_id = "HuggingFaceH4/zephyr-7b-beta"
arch = "mistral"

[anymoe]
dataset_json = "examples/amoe.json"
prefix = "model.layers"
mlp = "mlp"
model_ids = ["EricB/example_adapter"]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[anymoe.config]
hidden_size = 4096

[anymoe.config.expert_type.lora_adapter]
rank = 16
alpha = 16
target_modules = ["gate_proj"]
```

## 示例

## `mistralrs-server`

CLI 使用通过 [TOML 选择器](TOML_SELECTOR.md#anymoe) 进行，您也可以在其中找到所需字段的文档。

例如，使用演示微调专家：
```
./mistralrs-server -i toml -f toml-selectors/anymoe.toml
```

使用演示 LoRA 专家：
```
./mistralrs-server -i toml -f toml-selectors/anymoe_lora.toml
```

## Python 示例
```py
from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    AnyMoeConfig,
    AnyMoeExpertType,
)

runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
    ),
    anymoe_config=AnyMoeConfig(
        hidden_size=4096,
        dataset_json="examples/amoe.json",
        prefix="model.layers",
        mlp="mlp",
        expert_type=AnyMoeExpertType.FineTuned(),
        lr=1e-3,
        epochs=100,
        batch_size=4,
        model_ids=["HuggingFaceH4/zephyr-7b-beta"],
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

## Rust API
您可以在此处找到此示例 [../mistralrs/examples/anymoe/main.rs](../mistralrs/examples/anymoe/main.rs)。

```rust
use anyhow::Result;
use mistralrs::{
    AnyMoeConfig, AnyMoeExpertType, AnyMoeModelBuilder, IsqType, PagedAttentionMetaBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let text_builder = TextModelBuilder::new("mistralai/Mistral-7B-Instruct-v0.1")
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?;

    let model = AnyMoeModelBuilder::from_text_builder(
        text_builder,
        AnyMoeConfig {
            hidden_size: 4096,
            lr: 1e-3,
            epochs: 100,
            batch_size: 4,
            expert_type: AnyMoeExpertType::LoraAdapter {
                rank: 64,
                alpha: 16.,
                target_modules: vec!["gate_proj".to_string()],
            },
            gate_model_id: None, // 设置为 Some("path/to/model/id") 以获取预训练门控模型 ID
            training: true,
            loss_csv_path: None,
        },
        "model.layers",
        "mlp",
        "examples/amoe.json",
        vec!["HuggingFaceH4/zephyr-7b-beta"],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
    .build()
    .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}

```

## MOE LFU 缓存支持

AnyMoE 模型自动受益于 **MOE LFU 缓存** 机制，该机制优化专家选择并提高推理性能。

### 如何与 AnyMoE 配合使用
1. **自动专家跟踪**：记录推理过程中每个专家的访问频率
2. **智能缓存**：将频繁使用的专家保持在随时可用状态
3. **自适应优化**：适应您特定的 AnyMoE 配置
4. **无缝集成**：无需额外配置

### AnyMoE 的优势
- **更快的推理**：减少专家加载开销
- **更低的内存使用**：根据实际专家访问模式优化缓存使用
- **更好的可扩展性**：更好地处理具有许多专家的模型
- **自动**：默认启用，无需手动设置

### 使用方法
MOE LFU 缓存自动为所有 AnyMoE 模型启用。只需照常运行 AnyMoE 模型：

```bash
./mistralrs-server -i toml -f toml-selectors/anymoe.toml
```

此 LFU 缓存实现与微调专家和 LoRA 适配器专家无缝配合，为所有 AnyMoE 配置提供更高的性能。