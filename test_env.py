print('===== 验证 SanText 环境 =====')

# 1. PyTorch
import torch
print(f'✅ torch: {torch.__version__}')
print(f'   MPS可用(Apple GPU): {torch.backends.mps.is_available()}')

# 2. Transformers
import transformers
print(f'✅ transformers: {transformers.__version__}')

# 3. NumPy
import numpy
print(f'✅ numpy: {numpy.__version__}')

# 4. SciPy
import scipy
print(f'✅ scipy: {scipy.__version__}')

# 5. Scikit-learn
import sklearn
print(f'✅ scikit-learn: {sklearn.__version__}')

# 6. spaCy + 英文模型
import spacy
print(f'✅ spacy: {spacy.__version__}')
nlp = spacy.load('en_core_web_sm')
doc = nlp('Hello world')
print(f'   分词测试: {[token.text for token in doc]}')

# 7. tqdm
import tqdm
print(f'✅ tqdm: {tqdm.__version__}')

# 8. datasets
import datasets
print(f'✅ datasets: {datasets.__version__}')

# 9. filelock
import filelock
print(f'✅ filelock: {filelock.__version__}')

# 10. 项目核心模块
import sys
sys.path.insert(0, '/Users/yyr/dp_prompt/SanText-main')
from SanText import cal_probability, SanText, get_sanitized_doc
print(f'✅ SanText 核心模块导入成功')

print()
print('===== 🎉 所有依赖验证通过！环境配置完成 =====')
