# 支持 MiMo-V2-Flash 模型 GGUF 量化推理

## 实现计划

### 1. 添加 MiMo-V2-Flash 到 GGUFArchitecture 枚举
- **文件**: `mistralrs-core/src/gguf/mod.rs`
- **操作**: 在 `GGUFArchitecture` 枚举中添加 `MiMoV2Flash` 变体

### 2. 创建量化模型实现文件
- **文件**: `mistralrs-core/src/models/quantized_mimov2_flash.rs`
- **操作**: 
  - 实现 `ModelWeights` 结构体，包含 MiMo-V2-Flash 模型的量化层
  - 实现 `FromGGUF` trait 用于从 GGUF 文件加载模型
  - 实现 MOE 层的 GGUF 量化支持
  - 实现前向传播方法

### 3. 更新 GGUF 管道支持
- **文件**: `mistralrs-core/src/pipeline/gguf.rs`
- **操作**: 
  - 在 `Model` 枚举中添加 `MiMoV2Flash` 变体
  - 在 `load_model_from_path` 方法中添加 MiMo-V2-Flash 模型的加载逻辑
  - 添加 MiMo-V2-Flash 模型的前向传播支持

### 4. 更新导入和匹配语句
- **文件**: `mistralrs-core/src/pipeline/gguf.rs`
- **操作**: 
  - 添加 `quantized_mimov2_flash` 模块的导入
  - 在模型创建匹配语句中添加 MiMo-V2-Flash 分支
  - 在前向传播匹配语句中添加 MiMo-V2-Flash 分支

## 实现细节

### 量化模型结构
- 参考 `quantized_qwen3_moe.rs` 的实现
- 使用 GGUF 特定的张量加载方法
- 实现 MOE 层的量化支持
- 支持前向传播和注意力机制

### GGUF 元数据处理
- 提取 MiMo-V2-Flash 模型的特定配置参数
- 支持模型架构自动检测
- 处理 MOE 相关的元数据

### 模型加载流程
1. 从 GGUF 文件读取元数据
2. 验证模型架构为 MiMo-V2-Flash
3. 加载嵌入层、归一化层和输出层
4. 加载每个解码器层，包括注意力和 MOE 组件
5. 初始化旋转嵌入和缓存

## 测试计划
- 确保模型能够从 GGUF 文件成功加载
- 验证前向传播能够正常执行
- 测试量化推理的正确性
- 确保与现有 GGUF 功能兼容

## 预期结果

完成后，用户将能够使用以下命令运行 MiMo-V2-Flash 的 GGUF 量化模型：
```bash
./mistralrs-server -i gguf -m path/to/mimov2-flash-model -f model-quant.gguf
```