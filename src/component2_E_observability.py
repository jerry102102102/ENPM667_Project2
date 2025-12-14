import sympy as sp

# ----------------------------------------------------------------------
# Parameters (symbols)
# ----------------------------------------------------------------------
M, m1, m2, l1, l2, g = sp.symbols('M m1 m2 l1 l2 g', positive=True)

# ----------------------------------------------------------------------
# Linearized A, B around x = 0, θ1 = 0, θ2 = 0
# State: x = [x, xdot, θ1, θ1dot, θ2, θ2dot]^T
# ----------------------------------------------------------------------
A = sp.Matrix([
    [0, 1, 0, 0, 0, 0],
    [0, 0, g*m1/M, 0, g*m2/M, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, -g*(1 + m1/M)/l1, 0, -g*(m2/(M*l1)), 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, -g*(m1/(M*l2)), 0, -g*(1 + m2/M)/l2, 0],
])

# input: u = F
B = sp.Matrix([
    [0],
    [1/M],
    [0],
    [-1/(M*l1)],
    [0],
    [-1/(M*l2)]
])

# ----------------------------------------------------------------------
# Candidate output choices (each defines a C matrix)
# y1 = x(t)
# y2 = (θ1, θ2)
# y3 = (x, θ2)
# y4 = (x, θ1, θ2)
# ----------------------------------------------------------------------
C_list = []

# Case 1: y = x (cart position only)
C1 = sp.Matrix([[1, 0, 0, 0, 0, 0]])
C_list.append(("y = x", C1))

# Case 2: y = [θ1, θ2]^T
C2 = sp.Matrix([
    [0, 0, 1, 0, 0, 0],   # θ1
    [0, 0, 0, 0, 1, 0]    # θ2
])
C_list.append(("y = (θ1, θ2)", C2))

# Case 3: y = [x, θ2]^T
C3 = sp.Matrix([
    [1, 0, 0, 0, 0, 0],   # x
    [0, 0, 0, 0, 1, 0]    # θ2
])
C_list.append(("y = (x, θ2)", C3))

# Case 4: y = [x, θ1, θ2]^T
C4 = sp.Matrix([
    [1, 0, 0, 0, 0, 0],   # x
    [0, 0, 1, 0, 0, 0],   # θ1
    [0, 0, 0, 0, 1, 0]    # θ2
])
C_list.append(("y = (x, θ1, θ2)", C4))

# ----------------------------------------------------------------------
# Controllability matrix for a pair (A,B)
# (here we會用在「對偶系統」(A^T, C^T) 上)
# ----------------------------------------------------------------------
def controllability_matrix(A, B):
    n = A.shape[0]
    Co = B
    for i in range(1, n):
        Co = Co.row_join(A**i * B)
    return Co

# ----------------------------------------------------------------------
# 使用對偶方法檢查 observability:
# (A, C) observable  <=>  (A^T, C^T) controllable
# ----------------------------------------------------------------------
print("Symbolic observability test via duality (A^T, C^T):\n")

for name, C in C_list:
    A_dual = A.T
    B_dual = C.T   # 在對偶系統裡 C^T 扮演 "B"
    Co_dual = controllability_matrix(A_dual, B_dual)
    rank_Co = Co_dual.rank()

    print(f"Output choice: {name}")
    print("  C = ")
    sp.pprint(C)
    print(f"  rank(Controllability(A^T, C^T)) = {rank_Co}")

    # 判斷 observable (state 維度是 6)
    if rank_Co == 6:
        print("  ==> (A, C) is observable for this output choice.\n")
    else:
        print("  ==> (A, C) is NOT observable for this output choice.\n")

# ----------------------------------------------------------------------
# (選擇性) 代入具體的數值來確認 rank，不想用可以刪掉這一段
# M = 1000 kg, m1 = m2 = 100 kg, l1 = 20 m, l2 = 10 m, g = 9.81 m/s^2
# ----------------------------------------------------------------------
subs_vals = {
    M: 1000.0,
    m1: 100.0,
    m2: 100.0,
    l1: 20.0,
    l2: 10.0,
    g: 9.81
}

print("\nNumeric check with M=1000, m1=m2=100, l1=20, l2=10, g=9.81:\n")

A_num = A.subs(subs_vals)

for name, C in C_list:
    C_num = C.subs(subs_vals)
    A_dual_num = A_num.T
    B_dual_num = C_num.T

    Co_dual_num = controllability_matrix(A_dual_num, B_dual_num)
    rank_Co_num = Co_dual_num.rank()

    print(f"Output choice: {name}")
    print(f"  numeric rank(Controllability(A^T, C^T)) = {rank_Co_num}")
    if rank_Co_num == 6:
        print("  ==> Numerically observable (full rank 6).\n")
    else:
        print("  ==> Numerically NOT observable.\n")
