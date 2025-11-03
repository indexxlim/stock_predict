"""
특성 중요도 분석 - 핵심 특성과 유망 특성 선별
- 상관관계 분석
- 모델 기반 Feature Importance
- SHAP values 분석
- Mutual Information
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIG ============
DATA_PATH = Path('.')
RANDOM_STATE = 42

print("=" * 80)
print("특성 중요도 분석")
print("=" * 80)

# ============ 데이터 로드 ============
print("\n[1단계: 데이터 로드]")
train = (
    pl.read_csv(DATA_PATH / "train.csv")
    .rename({'market_forward_excess_returns': 'target'})
    .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
    .head(-10)
)

print(f"Train shape: {train.shape}")

# ============ 기본 특성 추출 ============
print("\n[2단계: 기본 특성 분석]")

# 특성 그룹별로 분류
feature_groups = {
    'D': [col for col in train.columns if col.startswith('D')],
    'E': [col for col in train.columns if col.startswith('E')],
    'I': [col for col in train.columns if col.startswith('I')],
    'M': [col for col in train.columns if col.startswith('M')],
    'P': [col for col in train.columns if col.startswith('P')],
    'S': [col for col in train.columns if col.startswith('S')],
    'V': [col for col in train.columns if col.startswith('V')]
}

# 모든 특성 컬럼
all_features = []
for cols in feature_groups.values():
    all_features.extend(cols)

print(f"전체 특성 수: {len(all_features)}")

# ============ 방법 1: 상관관계 분석 ============
print("\n[3단계: 상관관계 분석]")

# 결측치를 중앙값으로 채우기
train_filled = train.to_pandas()
for col in all_features:
    if col in train_filled.columns:
        train_filled[col].fillna(train_filled[col].median(), inplace=True)

# 타겟과의 상관관계 계산
correlations = {}
for feature in all_features:
    if feature in train_filled.columns:
        corr = train_filled[feature].corr(train_filled['target'])
        correlations[feature] = abs(corr)  # 절대값으로 저장

# 상관관계로 정렬
corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Abs_Correlation'])
corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

print("\n타겟과의 상관관계 상위 20개 특성:")
print(corr_df.head(20).to_string())

# ============ 방법 2: Mutual Information ============
print("\n[4단계: Mutual Information 분석]")

X = train_filled[all_features].values
y = train_filled['target'].values

# Mutual Information 계산
mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE, n_neighbors=5)
mi_df = pd.DataFrame({
    'Feature': all_features,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\nMutual Information 상위 20개 특성:")
print(mi_df.head(20).to_string())

# ============ 방법 3: ElasticNet Feature Importance ============
print("\n[5단계: ElasticNet 특성 중요도]")

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ElasticNet CV
enet = ElasticNetCV(
    l1_ratio=0.5,
    cv=10,
    alphas=np.logspace(-4, 2, 100),
    max_iter=1000000,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
enet.fit(X_scaled, y)

# 계수의 절대값을 중요도로 사용
enet_importance = np.abs(enet.coef_)
enet_df = pd.DataFrame({
    'Feature': all_features,
    'ElasticNet_Importance': enet_importance
}).sort_values('ElasticNet_Importance', ascending=False)

print(f"\nElasticNet 최적 alpha: {enet.alpha_:.6f}")
print("\nElasticNet 특성 중요도 상위 20개:")
print(enet_df.head(20).to_string())

# ============ 방법 4: LightGBM Feature Importance ============
print("\n[6단계: LightGBM 특성 중요도]")

# LightGBM 학습
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_scaled, y)
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500
)

# Feature importance
lgb_importance = lgb_model.feature_importance(importance_type='gain')
lgb_df = pd.DataFrame({
    'Feature': all_features,
    'LightGBM_Importance': lgb_importance
}).sort_values('LightGBM_Importance', ascending=False)

print("\nLightGBM 특성 중요도 상위 20개:")
print(lgb_df.head(20).to_string())

# ============ 종합 분석 ============
print("\n[7단계: 종합 분석]")

# 모든 방법을 결합
combined_df = corr_df.copy()
combined_df = combined_df.merge(mi_df[['Feature', 'MI_Score']], on='Feature', how='left')
combined_df = combined_df.merge(enet_df[['Feature', 'ElasticNet_Importance']], on='Feature', how='left')
combined_df = combined_df.merge(lgb_df[['Feature', 'LightGBM_Importance']], on='Feature', how='left')

# 각 방법의 순위 계산
combined_df['Corr_Rank'] = combined_df['Abs_Correlation'].rank(ascending=False)
combined_df['MI_Rank'] = combined_df['MI_Score'].rank(ascending=False)
combined_df['Enet_Rank'] = combined_df['ElasticNet_Importance'].rank(ascending=False)
combined_df['LGB_Rank'] = combined_df['LightGBM_Importance'].rank(ascending=False)

# 평균 순위 계산
combined_df['Average_Rank'] = combined_df[['Corr_Rank', 'MI_Rank', 'Enet_Rank', 'LGB_Rank']].mean(axis=1)
combined_df = combined_df.sort_values('Average_Rank')

print("\n종합 특성 중요도 상위 30개 (평균 순위 기준):")
print(combined_df.head(30)[['Feature', 'Average_Rank', 'Abs_Correlation', 'MI_Score', 'ElasticNet_Importance', 'LightGBM_Importance']].to_string())

# ============ 그룹별 분석 ============
print("\n[8단계: 그룹별 특성 중요도]")

for group_name, group_features in feature_groups.items():
    group_df = combined_df[combined_df['Feature'].isin(group_features)]
    if len(group_df) > 0:
        print(f"\n{group_name} 그룹 상위 5개:")
        print(group_df.head(5)[['Feature', 'Average_Rank']].to_string(index=False))

# ============ 최종 추천 특성 ============
print("\n" + "=" * 80)
print("최종 추천 특성")
print("=" * 80)

# 1위 노트북의 특성 (참고)
top_notebook_features = ["S2", "E2", "E3", "P9", "S1", "S5", "I2", "P8", "P10", "P12", "P13"]

# 평균 순위 기준 상위 15개
top_15_by_avg = combined_df.head(15)['Feature'].tolist()

# ElasticNet 기준 상위 15개 (계수가 0이 아닌 것만)
top_15_by_enet = enet_df[enet_df['ElasticNet_Importance'] > 0].head(15)['Feature'].tolist()

# LightGBM 기준 상위 15개
top_15_by_lgb = lgb_df.head(15)['Feature'].tolist()

print("\n1. 1위 노트북 특성 (13개):")
print(top_notebook_features)

print("\n2. 평균 순위 기준 상위 15개:")
print(top_15_by_avg)

print("\n3. ElasticNet 기준 상위 15개 (비영계수):")
print(top_15_by_enet)

print("\n4. LightGBM 기준 상위 15개:")
print(top_15_by_lgb)

# 추천 핵심 특성 (여러 방법에서 공통적으로 높은 순위)
core_features = list(set(top_15_by_avg[:10]) & set(top_15_by_enet[:10]) & set(top_15_by_lgb[:10]))
print(f"\n5. 추천 핵심 특성 ({len(core_features)}개) - 3가지 방법 모두 상위 10위:")
print(core_features)

# 유망 추가 특성 (상위 30개에서 핵심 특성 제외)
additional_features = [f for f in combined_df.head(30)['Feature'].tolist() if f not in core_features]
print(f"\n6. 유망 추가 특성 ({len(additional_features)}개):")
print(additional_features)

# ============ 결과 저장 ============
combined_df.to_csv('feature_importance_analysis.csv', index=False)
print("\n분석 결과가 'feature_importance_analysis.csv'에 저장되었습니다.")

# ============ 시각화 ============
print("\n[9단계: 시각화 생성]")

# 상위 20개 특성의 중요도 비교
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Correlation
top_20 = corr_df.head(20)
axes[0, 0].barh(range(len(top_20)), top_20['Abs_Correlation'])
axes[0, 0].set_yticks(range(len(top_20)))
axes[0, 0].set_yticklabels(top_20['Feature'])
axes[0, 0].set_xlabel('Absolute Correlation')
axes[0, 0].set_title('Top 20 Features by Correlation')
axes[0, 0].invert_yaxis()

# Mutual Information
top_20_mi = mi_df.head(20)
axes[0, 1].barh(range(len(top_20_mi)), top_20_mi['MI_Score'])
axes[0, 1].set_yticks(range(len(top_20_mi)))
axes[0, 1].set_yticklabels(top_20_mi['Feature'])
axes[0, 1].set_xlabel('MI Score')
axes[0, 1].set_title('Top 20 Features by Mutual Information')
axes[0, 1].invert_yaxis()

# ElasticNet
top_20_enet = enet_df.head(20)
axes[1, 0].barh(range(len(top_20_enet)), top_20_enet['ElasticNet_Importance'])
axes[1, 0].set_yticks(range(len(top_20_enet)))
axes[1, 0].set_yticklabels(top_20_enet['Feature'])
axes[1, 0].set_xlabel('ElasticNet Coefficient')
axes[1, 0].set_title('Top 20 Features by ElasticNet')
axes[1, 0].invert_yaxis()

# LightGBM
top_20_lgb = lgb_df.head(20)
axes[1, 1].barh(range(len(top_20_lgb)), top_20_lgb['LightGBM_Importance'])
axes[1, 1].set_yticks(range(len(top_20_lgb)))
axes[1, 1].set_yticklabels(top_20_lgb['Feature'])
axes[1, 1].set_xlabel('LightGBM Gain')
axes[1, 1].set_title('Top 20 Features by LightGBM')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("시각화가 'feature_importance_comparison.png'에 저장되었습니다.")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
