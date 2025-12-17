# Phi 3.5 模型：[`microsoft/Phi-3.5-MoE-instruct`](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

Phi 3.5 MoE 模型是一个 16x3.8B 参数的解码器-only 文本到文本混合专家 LLM。

- **128k tokens** 的上下文长度
- 在 **4.9T tokens** 上训练
- 16 个专家（16x3.8B 参数），**6.6B 活跃参数**
- 预期推理性能相当于 7B 模型

## 关于 MoE 机制
1) 计算路由门控 logits
2) 从路由门控 logits 中，选择前 2 个选中的专家和相关权重
3) 序列中每个 token 的隐藏状态通过（如果被选中）将专家输出应用到该 token，然后加权计算。
    - 如果为 token 选择了多个专家，则这会成为加权和
    - 设计灵活：可以选择 2 个或 1 个专家，支持密集或稀疏门控

## MOE LFU 缓存

Mistral.rs 为 MOE 模型实现了 **最少使用频率（LFU）** 缓存机制，以优化专家选择并提高推理性能。

### 工作原理
1. **专家访问跟踪**：记录推理过程中每个专家的访问频率
2. **最后访问时间**：跟踪每个专家的最后使用时间
3. **智能缓存**：根据访问模式优化专家加载和使用
4. **自适应优化**：根据实际使用数据调整缓存策略

### 优势
- **更快的推理**：通过将频繁使用的专家保持在随时可用状态，减少专家加载开销
- **更低的内存使用**：基于实际使用情况优化专家缓存，高效使用内存
- **更好的可扩展性**：更好地处理具有更多专家的更大 MOE 模型
- **自动**：默认启用所有 MOE 模型，无需手动配置

### 支持的 MOE 模型
- Phi 3.5 MoE
- Qwen 3 MoE
- DeepSeek V3 MoE
- Mixtral 8x7B
- 任何其他具有 `moe_num_experts` 配置的 MOE 模型

### 使用方法
MOE LFU 缓存默认情况下对所有 MOE 模型启用。只需照常运行 MOE 模型，LFU 缓存将自动优化专家选择：

```bash
./mistralrs-server --isq 4 -i plain -m microsoft/Phi-3.5-MoE-instruct
```

### 技术细节
- **专家频率跟踪**：为每个专家的访问频率维护计数器
- **基于时间的淘汰**：同时考虑访问频率和最近访问时间
- **自适应缩放**：适应不同的模型大小和专家数量
- **低开销**：跟踪本身的性能影响最小

这种 LFU 缓存实现设计为与所有 MOE 模型无缝配合，无需用户进行任何额外配置即可提供更高的性能。

```
./mistralrs-server --isq 4 -i plain -m microsoft/Phi-3.5-MoE-instruct
```

> [!NOTE]
> 该模型支持 MoQE，可以在各种 API 中的 ISQ 组织参数中激活，如下所示：

```
./mistralrs-server --isq 4 -i plain -m microsoft/Phi-3.5-MoE-instruct --organization moqe
```

## HTTP API

```
./mistralrs-server --isq 4 --port 1234 plain -m microsoft/Phi-3.5-MoE-instruct
```

```py
import openai

messages = []
prompt = input("输入系统提示 >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})


while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
    resp = completion.choices[0].message.content
    print(resp)
    messages.append({"role": "assistant", "content": resp})
```

## Python API
```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-MoE-instruct",
        arch=Architecture.Phi3_5MoE ,
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
您可以在此处找到此示例 [../mistralrs/examples/phi3_5_moe/main.rs](../mistralrs/examples/phi3_5_moe/main.rs)。

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-MoE-instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
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