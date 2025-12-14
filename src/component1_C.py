import sympy as sp

# 1. 定義符號參數
M, m1, m2, l1, l2, g = sp.symbols('M m1 m2 l1 l2 g', nonzero=True)

# 2. 線性化後的 A, B 矩陣（跟我們在 B 題算出來的一樣）
A = sp.Matrix([
    [0, 1,                     0, 0,                     0, 0],
    [0, 0,              g*m1/M, 0,              g*m2/M, 0],
    [0, 0,                     0, 1,                     0, 0],
    [0, 0, -g*(M + m1)/(M*l1), 0,       -g*m2/(M*l1), 0],
    [0, 0,                     0, 0,                     0, 1],
    [0, 0,       -g*m1/(M*l2), 0, -g*(M + m2)/(M*l2), 0]
])

B = sp.Matrix([
    [0],
    [1/M],
    [0],
    [-1/(M*l1)],
    [0],
    [-1/(M*l2)]
])

# 3. 建 controllability matrix C = [B, AB, A^2B, ... , A^{5}B]
n = A.shape[0]  # n = 6
blocks = [B]
for k in range(1, n):
    blocks.append(A**k * B)

C = sp.Matrix.hstack(*blocks)

print("Controllability matrix C:")
sp.pprint(C)

# 4. 算 rank
rank_C = C.rank()
print("\nrank(C) =", rank_C)

# 5. 算 det(C) 並化簡、因式分解
det_C = sp.factor(sp.simplify(C.det()))
print("\ndet(C) =")
sp.pprint(det_C)

# 6. （選配）帶一組數值檢查 rank 真的 = 6
subs_example = {
    M: 10.0,
    m1: 1.0,
    m2: 2.0,
    l1: 1.0,
    l2: 1.5,
    g: 9.81
}
rank_numeric = C.subs(subs_example).rank()
print("\nNumeric rank with example parameters =", rank_numeric)
