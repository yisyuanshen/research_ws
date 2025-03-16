from controller import Supervisor
import numpy as np
import osqp
from scipy import sparse

m = 22
gravity = 9.81
dt = 0.001
N = 10

A = np.array([[1, 0, 0, dt, 0, 0],
              [0, 1, 0, 0, dt, 0],
              [0, 0, 1, 0, 0, dt],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

B = np.array([[0,    0,    0],
              [0,    0,    0],
              [0,    0,    0],
              [dt/m, 0,    0],
              [0,    dt/m, 0],
              [0,    0,    dt/m]])

n_x = A.shape[0]
n_u = B.shape[1]

A_qp = sparse.vstack([sparse.csc_matrix(np.linalg.matrix_power(A, i + 1)) for i in range(N)])
B_qp = sparse.vstack([
    sparse.hstack([
        sparse.csc_matrix(np.linalg.matrix_power(A, i - j) @ B) if j <= i else 
        sparse.csc_matrix((n_x, n_u)) for j in range(N - 1)
    ]) for i in range(N)])

Q = np.diag([1e9, 1e9, 1e9, 5*1e5, 5*1e5, 5*1e5])
R = np.diag([1, 1, 1])

Q_N = sparse.block_diag([Q] * N, format='csc')
R_N = sparse.block_diag([R] * (N - 1), format='csc')


def model_predictive_control(x, x_ref):
    H = B_qp.T @ Q_N @ B_qp + R_N
    g = 2 * ((A_qp @ x).reshape(-1, 1) - x_ref.reshape(-1, 1)).T @ Q_N @ B_qp

    # Constraints on control inputs (not states)
    u_max = 300.0
    u_min = -300.0
    lb = np.full(((N - 1) * n_u,), u_min)  # Ensuring correct dimensions
    ub = np.full(((N - 1) * n_u,), u_max)  # Ensuring correct dimensions

    # Define equality constraints as an empty matrix if no constraints exist
    A_eq = sparse.csc_matrix(((N - 1) * n_u, (N - 1) * n_u))  # Ensure proper shape
    b_eq = np.zeros((N - 1) * n_u)  # Ensure correct size

    # Solve QP with OSQP
    qp = osqp.OSQP()
    qp.setup(P=sparse.csc_matrix(H), q=g.flatten(), A=A_eq, l=lb, u=ub, verbose=True)  # Ensure H is sparse

    # Solve
    res = qp.solve()

    # Extract optimal control inputs
    u_opt = res.x.reshape(N - 1, n_u)
    print("Optimal control sequence:", u_opt)

    return u_opt[0]


def main():
    supervisor = Supervisor()
    robot = supervisor.getFromDef("robot")
    
    timestep = int(supervisor.getBasicTimeStep())
    
    loop_count = 0
    while supervisor.step(timestep) != -1:
        # Update robot state
        robot_pos = np.array(robot.getPosition())
        robot_ang = np.array(robot.getOrientation()).reshape(3, 3)
        robot_vel = np.array(robot.getVelocity())
        
        # print("Robot Ang:", robot_ang)
        # print("Robot Ang Vel:", robot_vel[3:])
        # print("- - -")
        
        # Current state and reference state
        x = np.array([robot_pos[0], robot_pos[1], robot_pos[2], robot_vel[0], robot_vel[1], robot_vel[2]])
        x_ref = np.array([[loop_count/1000, 1, 0.5, 0, 0, 0]]*N).flatten()
        
        # Compute optimal force
        force = model_predictive_control(x, x_ref)
        
        # Apply force (world frame)
        robot.addForce(force.tolist(), False)
        
        print("- - -")
        print("Robot Pos:", robot_pos)
        print("Robot Vel:", robot_vel[:3])
        print("Applied Force:", force)
        print("= = = = =")
        
        loop_count += 1


if __name__ == "__main__":
    main()
