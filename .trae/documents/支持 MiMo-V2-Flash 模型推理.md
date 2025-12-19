# 支持 MiMo-V2-Flash 模型推理实现计划

## 1. 模型类型定义与配置

### 1.1 添加 MiMo-V2-Flash 模型类型
- **文件**: `mistralrs-core/src/pipeline/loaders/normal_loaders.rs`
- **内容**: 在 `NormalLoaderType` 枚举中添加 `MiMoV2Flash` 变体
- **配置映射**: 将 Hugging Face 模型名称映射到新的模型类型

### 1.2 创建 MiMo-V2-Flash 配置结构体
- **文件**: `mistralrs-core/src/models/mimov2_flash.rs`
- **内容**: 定义 `Config` 结构体，包含 MOE 相关配置:
  - `vocab_size`
  - `hidden_size`
  - `intermediate_size`
  - `num_hidden_layers`
  - `num_attention_heads`
  - `num_key_value_heads`
  - `num_experts` (MOE 特有)
  - `num_experts_per_tok` (MOE 特有)
  - `moe_intermediate_size` (MOE 特有)
  - 其他标准 transformer 配置

## 2. 模型实现

### 2.1 创建 MiMo-V2-Flash 模型实现
- **文件**: `mistralrs-core/src/models/mimov2_flash.rs`
- **内容**: 
  - 实现 `Model` 结构体，包含嵌入层、解码器层、归一化层等
  - 实现 `NormalModel` trait
  - 实现 `AnyMoeBaseModelMixin` trait
  - 集成现有的 `MoEExperts` 系统
  - 支持 LFU 缓存机制

### 2.2 实现解码器层
- **文件**: `mistralrs-core/src/models/mimov2_flash.rs`
- **内容**: 实现包含注意力机制和 MOE 专家层的解码器层
  - `Attention` 结构体
  - `DecoderLayer` 结构体
  - 集成 `MoEExperts` 进行专家推理

## 3. 模型加载器实现

### 3.1 创建 MiMo-V2-Flash 加载器
- **文件**: `mistralrs-core/src/pipeline/loaders/normal_loaders.rs`
- **内容**: 实现 `MiMoV2FlashLoader` 结构体，实现 `NormalModelLoader` trait
  - 解析模型配置
  - 加载模型权重
  - 初始化 `MoEExperts` 系统
  - 配置 LFU 缓存

### 3.2 更新模型名称映射
- **文件**: `mistralrs-core/src/pipeline/loaders/normal_loaders.rs`
- **内容**: 在 `from_causal_lm_name` 方法中添加 MiMo-V2-Flash 模型名称映射

## 4. 测试与验证

### 4.1 模型配置测试
- **测试**: 验证模型配置正确解析
- **内容**: 确保 MOE 相关参数正确加载

### 4.2 模型加载测试
- **测试**: 验证模型权重正确加载
- **内容**: 确保 `MoEExperts` 系统正确初始化

### 4.3 推理测试
- **测试**: 验证模型能够进行正常推理
- **内容**: 运行简单的推理测试，确保输出符合预期

## 5. 文档更新

### 5.1 更新支持的模型列表
- **文件**: `README.md`
- **内容**: 添加 MiMo-V2-Flash 到支持的模型列表

### 5.2 添加模型文档
- **文件**: `docs/MIMOV2FLASH.md`
- **内容**: 编写 MiMo-V2-Flash 模型的使用文档，包括:
  - 模型介绍
  - 支持的配置
  - 使用示例
  - MOE LFU 缓存支持

## 技术要点

1. **重用现有 MOE 架构**: 利用已有的 `MoEExperts`、`ExpertUsageTracker` 和 `ExpertCacheManager` 系统
2. **正确配置 MOE 参数**: 确保 MiMo-V2-Flash 的 MOE 参数与现有系统兼容
3. **支持 LFU 缓存**: 启用 MOE LFU 缓存机制，优化推理性能
4. **遵循现有代码风格**: 保持与项目现有代码一致的风格和结构
5. **确保跨平台兼容性**: 支持 CUDA、Metal 和 CPU 推理

## 预期结果

完成后，用户将能够使用以下命令运行 MiMo-V2-Flash 模型推理:
```bash
./mistralrs-server -i run -m ../models/MiMo-V2-Flash/
```

模型将正确加载，并利用现有的 MOE 专家系统和 LFU 缓存机制进行高效推理。