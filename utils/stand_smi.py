import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 读取 CSV 文件
file_path = 'data/ET30.csv'  # 文件路径
df = pd.read_csv(file_path)

# 确保 SMILES 列存在
if 'SMILES' not in df.columns:
    raise ValueError("CSV 文件中未找到 'SMILES' 列")

# 定义 SMILES 标准化函数
def standardize_smiles(smiles):
    try:
        # 转换为 RDKit 分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 标准化：去除立体化学信息，规范化分子
        mol = Chem.RemoveHs(mol)  # 移除氢原子
        Chem.RemoveStereochemistry(mol)  # 移除立体化学信息
        # 转换为标准 SMILES
        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return standardized_smiles
    except:
        return None

# 记录无效 SMILES 的行号（基于 CSV 的行号，从 2 开始，因为第 1 行是标题）
invalid_rows = []
for idx, smiles in enumerate(df['SMILES'], start=2):
    if standardize_smiles(smiles) is None:
        invalid_rows.append(idx)

# 打印无效 SMILES 的行号
if invalid_rows:
    print("以下行号的 SMILES 解析错误：")
    for row in invalid_rows:
        print(f"行 {row}: {df.loc[row-2, 'SMILES']}")
else:
    print("没有发现无效的 SMILES。")

# 删除包含无效 SMILES 的行
df_cleaned = df[df['SMILES'].apply(standardize_smiles).notnull()].copy()

# 应用标准化函数到 SMILES 列
df_cleaned['Standardized_SMILES'] = df_cleaned['SMILES'].apply(standardize_smiles)

# 保存清洗和标准化后的结果到新的 CSV 文件
output_path = 'data/ET30_standardized_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"清洗并标准化后的 SMILES 已保存到 {output_path}")
print(f"共删除 {len(df) - len(df_cleaned)} 行无效 SMILES。")