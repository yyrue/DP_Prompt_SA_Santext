"""
测量词向量之间的最大距离，用于 MLDP 到标准 DP 的转换
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

def measure_max_distance(embedding_path, vocab_limit=10000):
    """测量词向量之间的最大距离"""
    
    print(f"加载词向量: {embedding_path}")
    embeddings = []
    words = []
    
    with open(embedding_path) as f:
        # 跳过可能的头部
        line = f.readline().rstrip().split(' ')
        if len(line) != 2:
            f.seek(0)
        
        for i, row in tqdm(enumerate(f), desc="读取词向量"):
            if i >= vocab_limit:
                break
            content = row.rstrip().split(' ')
            word = content[0]
            emb = [float(x) for x in content[1:]]
            embeddings.append(emb)
            words.append(word)
    
    embeddings = np.array(embeddings, dtype='f')
    print(f"词向量矩阵: {embeddings.shape}")
    
    # 计算距离矩阵
    print("计算距离矩阵...")
    distance_matrix = euclidean_distances(embeddings, embeddings)
    
    # 统计距离
    max_distance = distance_matrix.max()
    min_distance = distance_matrix[distance_matrix > 0].min()  # 排除对角线
    mean_distance = distance_matrix[distance_matrix > 0].mean()
    
    print("\n" + "="*50)
    print("距离统计结果")
    print("="*50)
    print(f"最大距离 (d_max): {max_distance:.4f}")
    print(f"最小非零距离: {min_distance:.4f}")
    print(f"平均距离: {mean_distance:.4f}")
    
    # 找到最远的词对
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    print(f"\n最远的词对:")
    print(f"  词1: {words[max_idx[0]]}")
    print(f"  词2: {words[max_idx[1]]}")
    print(f"  距离: {distance_matrix[max_idx]:.4f}")
    
    # 转换建议
    print("\n" + "="*50)
    print("MLDP -> 标准 DP 转换建议")
    print("="*50)
    print(f"如果目标: 标准 DP epsilon = 1")
    print(f"应设置: MLDP epsilon = 1 / {max_distance:.4f} = {1/max_distance:.4f}")
    print(f"\n如果目标: 标准 DP epsilon = 2")
    print(f"应设置: MLDP epsilon = 2 / {max_distance:.4f} = {2/max_distance:.4f}")
    print(f"\n如果目标: 标准 DP epsilon = 5")
    print(f"应设置: MLDP epsilon = 5 / {max_distance:.4f} = {5/max_distance:.4f}")
    
    print("\n" + "="*50)
    print("当前 SanText 设置分析 (epsilon=14)")
    print("="*50)
    effective_dp_epsilon = 14 * max_distance
    print(f"当前 MLDP epsilon = 14")
    print(f"等效标准 DP epsilon = 14 * {max_distance:.4f} = {effective_dp_epsilon:.4f}")
    print(f"→ 这是一个非常弱的隐私保证!")
    
    return max_distance

if __name__ == "__main__":
    embedding_path = "/home/youyaru/SanText-main/data/glove.840B.300d.txt"
    # 限制词汇量以加快计算（完整词表太大）
    max_distance = measure_max_distance(embedding_path, vocab_limit=5000)