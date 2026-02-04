import sys
import os
import torch
import types

# ==========================================
# 1. 解决 GOMP 冲突
# ==========================================
try:
    import graph_tool.all
except: pass

# ==========================================
# 2. 全能伪装系统 (修复 'cannot import utils')
# ==========================================
def create_fake_module(name):
    m = types.ModuleType(name)
    # 如果是包 (包含点号的父级)，需要设置 __path__
    if '.' not in name: 
        m.__path__ = []
    return m

# 核心：不仅注册 sys.modules，还要把子模块挂载到父模块上
def register_mock(name):
    parts = name.split('.')
    # 确保每一层都存在
    for i in range(1, len(parts) + 1):
        partial_name = '.'.join(parts[:i])
        
        # 如果尚未创建，则创建
        if partial_name not in sys.modules:
            m = create_fake_module(partial_name)
            sys.modules[partial_name] = m
        
        # 如果有父级，挂载到父级
        if i > 1:
            parent_name = '.'.join(parts[:i-1])
            child_name = parts[i-1]
            parent = sys.modules[parent_name]
            setattr(parent, child_name, sys.modules[partial_name])

# --- A. 注册所有可能的模块 ---
# 列表包含所有报错过的和可能报错的模块
modules_to_mock = [
    "datasets",
    "datasets.ppi_dataset",
    "src",
    "src.utils",               # <--- 修复你刚才的报错
    "src.datasets",
    "src.datasets.abstract_dataset",
    "src.datasets.spectre_dataset",
    "src.models",
    "src.models.transformer_model", 
    "src.diffusion",
    "src.diffusion.extra_features",
    "src.metrics",
    "src.metrics.abstract_metrics",
    "src.analysis"
]

print("⚡ 正在构建全息虚拟环境...")
for m in modules_to_mock:
    register_mock(m)

# --- B. 注入缺失的类定义 (解决 Can't get attribute) ---
def inject_class(module_name, class_name):
    if module_name not in sys.modules:
        register_mock(module_name)
    mod = sys.modules[module_name]
    # 创建一个带有正确 __module__ 的哑类
    cls = type(class_name, (object,), {"__module__": module_name})
    setattr(mod, class_name, cls)

# 注入 Dataset 相关的类
inject_class("datasets.ppi_dataset", "PPIDatasetInfos")
inject_class("datasets.ppi_dataset", "PPIDataModule")
inject_class("src.datasets.abstract_dataset", "AbstractDatasetInfos")
inject_class("src.datasets.abstract_dataset", "AbstractDataModule")
inject_class("src.utils", "PlaceHolder") # 防止 utils 里找某些类

# ==========================================
# 3. 诊断逻辑
# ==========================================
def inspect_checkpoint(checkpoint_path):
    print(f"\n📦 正在解剖: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print("❌ 文件不存在！")
        return

    try:
        # map_location='cpu'
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except AttributeError as e:
        print(f"\n❌ 属性缺失错误: {e}")
        print("👉 请检查脚本上方的 inject_class 部分，补充缺失的类名。")
        return
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        print("👉 可能是还有其他模块未 mock，请添加到 modules_to_mock 列表中。")
        return

    state_dict = ckpt.get('state_dict', {})
    print(f"✅ 读取成功！(State Dict 包含 {len(state_dict)} 个键)")
    
    print("\n" + "="*50)
    print("🕵️‍♂️ 侦探模式：检查输出层维度")
    print("="*50)
    
    # 目标维度
    TARGET_DIMS = [129, 128, 8, 5, 4]
    
    found_suspicious = False
    found_output_layer = False
    
    for key, val in state_dict.items():
        if not torch.is_tensor(val): continue
        if val.dim() < 2: continue 
        
        shape = tuple(val.shape)
        
        if any(d in TARGET_DIMS for d in shape):
            # 过滤非输出层
            if 'embedding' in key: continue 
            if 'attn' in key: continue      
            if 'norm' in key: continue      
            if 'time' in key: continue
            if 'input' in key: continue
            
            print(f"👉 发现层: {key:<50} | Shape: {shape}")
            
            out_dim = shape[0]
            
            if out_dim == 129:
                print(f"   ✨ [正确] 输出维度为 129。")
                found_output_layer = True
            elif out_dim in [8, 5, 4]:
                print(f"   💀 [异常] 输出维度为 {out_dim}。确认物理截断！")
                found_suspicious = True
                found_output_layer = True

    print("\n" + "-"*50)
    print("诊断结论：")
    if found_suspicious:
        print("🔴 **确认异常**：检测到维度为 8/5/4 的输出层。")
        print("   原因：旧缓存文件 (processed/ppi_train.pt) 导致模型构建错误。")
        print("   解决：执行 `rm -rf data/PPI/processed/` 并重新训练。")
    elif found_output_layer:
        print("🟢 **结构正常**：输出层维度为 129。")
        print("   如果生成结果仍为0，问题在于 Loss 权重或采样代码。")
    else:
        print("⚪ **未定**：未能自动定位输出层，请人工检查上方打印结果。")

CKPT_PATH = "/home/yyyy/codework/GARplus/DiGress/outputs/2026-01-23/11-15-51-ppi_gar/checkpoints/ppi_gar/last-v1.ckpt"

if __name__ == "__main__":
    inspect_checkpoint(CKPT_PATH)