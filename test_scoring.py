import sys
import os
sys.path.insert(0, 'd:/vs code/aliyun/demo')

# 修改scoring模块的函数来添加追踪
from agent import scoring

# 保存原始函数
original_score_action = scoring.score_action

# 创建包装函数来追踪调用
def traced_score_action(ctx, driver_id, cargo_id, action, params):
    print(f"✅ score_action called: driver_id={driver_id}, action={action}")
    result = original_score_action(ctx, driver_id, cargo_id, action, params)
    print(f"   Result: score={result.score}, feasible={result.feasible}, note={result.note}")
    return result

# 替换原始函数
scoring.score_action = traced_score_action

print("✅ scoring.py 模块已加载并准备追踪")
print(f"🔍 模块路径: {scoring.__file__}")
