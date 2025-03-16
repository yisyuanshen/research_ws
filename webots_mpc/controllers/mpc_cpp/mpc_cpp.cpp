#include <webots/Supervisor.hpp>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <iostream>

using namespace webots;
using namespace Eigen;

const int m = 22;
const double gravity = 9.81;
const double dt = 0.001;
const int N = 10;
const int n_x = 12;
const int n_u = 12;

MatrixXd A(n_x, n_x);
MatrixXd B(n_x, n_u);
MatrixXd Q(n_x, n_x);
MatrixXd R(n_u, n_u);

void initializeMatrices() {
    A << 1,   0,   0,   dt,  0,   0,   0,   0,   0,   0,   0,   0,
         0,   1,   0,   0,   dt,  0,   0,   0,   0,   0,   0,   0,
         0,   0,   1,   0,   0,   dt,  0,   0,   0,   0,   0,   0,
         0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   1,   0,   0,   dt,  0,   0,
         0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   dt,  0,
         0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   dt,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1;

    B << 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         dt/m,0,   0,   dt/m,0,   0,   dt/m,0,   0,   dt/m,0,   0,
         0,   dt/m,0,   0,   dt/m,0,   0,   dt/m,0,   0,   dt/m,0,
         0,   0,   dt/m,0,   0,   dt/m,0,   0,   dt/m,0,   0,   dt/m,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0;

    Q = MatrixXd::Zero(n_x, n_x);
    Q.diagonal() << 1e9, 1e9, 1e9, 5e5, 5e5, 5e5, 1e9, 1e9, 1e9, 5e5, 5e5, 5e5;
    
    R = MatrixXd::Identity(n_u, n_u);
}

VectorXd modelPredictiveControl(const VectorXd &x, const VectorXd &x_ref) {
    MatrixXd A_qp(N * n_x, n_x);
    MatrixXd B_qp(N * n_x, (N - 1) * n_u);
    
    A_qp.setZero();
    B_qp.setZero();

    // Build the prediction matrices
    for (int i = 0; i < N; ++i) {
        MatrixXd A_pow = MatrixXd::Identity(n_x, n_x);
        for (int k = 0; k < (i + 1); ++k) {
            A_pow *= A;
        }
        A_qp.block(i * n_x, 0, n_x, n_x) = A_pow;

        int max_j = (i < (N - 1)) ? (i + 1) : (N - 1);
        for (int j = 0; j < max_j; ++j) {
            MatrixXd A_temp = MatrixXd::Identity(n_x, n_x);
            for (int k = 0; k < (i - j); ++k) {
                A_temp *= A;
            }
            B_qp.block(i * n_x, j * n_u, n_x, n_u) = A_temp * B;
        }
    }

    // Assemble block diagonal Q and R matrices for the horizon.
    MatrixXd Q_N = MatrixXd::Zero(N * n_x, N * n_x);
    for (int i = 0; i < N; ++i) {
        Q_N.block(i * n_x, i * n_x, n_x, n_x) = Q;
    }

    MatrixXd R_N = MatrixXd::Zero((N - 1) * n_u, (N - 1) * n_u);
    for (int i = 0; i < (N - 1); ++i) {
        R_N.block(i * n_u, i * n_u, n_u, n_u) = R;
    }

    // Set up the QP cost function.
    MatrixXd H = 2 * (B_qp.transpose() * Q_N * B_qp + R_N);
    VectorXd g = 2 * B_qp.transpose() * Q_N * (A_qp * x - x_ref);

    // Regularize H to ensure positive definiteness.
    H += 1e-6 * MatrixXd::Identity(H.rows(), H.cols());

    OsqpEigen::Solver solver;
    // solver.settings()->setVerbosity(true);
    // solver.settings()->setWarmStart(true);
    // solver.settings()->setPolish(true);
    // solver.settings()->setLinearSystemSolver(QDLDL_SOLVER);
    // solver.settings()->setAlpha(1.6);

    int n_vars = (N - 1) * n_u;
    solver.data()->setNumberOfVariables(n_vars);
    solver.data()->setNumberOfConstraints(n_vars);

    SparseMatrix<double> H_sparse = H.sparseView();
    solver.data()->setHessianMatrix(H_sparse);
    solver.data()->setGradient(g);

    // Set the linear constraints matrix as identity for box constraints.
    SparseMatrix<double> A_constraint(n_vars, n_vars);
    A_constraint.setIdentity();
    solver.data()->setLinearConstraintsMatrix(A_constraint);

    VectorXd lb = VectorXd::Constant(n_vars, -300.0);
    VectorXd ub = VectorXd::Constant(n_vars, 300.0);
    solver.data()->setLowerBound(lb);
    solver.data()->setUpperBound(ub);

    // Initialize and solve the QP.
    solver.initSolver();
    solver.solveProblem();

    VectorXd u_opt = solver.getSolution();
    return u_opt.head(n_u);
}


int main() {
    Supervisor supervisor;
    Node *robot = supervisor.getFromDef("robot");
    initializeMatrices();
    int timestep = (int) supervisor.getBasicTimeStep();

    int loop_count = 0;
    while (supervisor.step(timestep) != -1) {
        const double *robot_pos = robot->getPosition();
        const double *robot_vel = robot->getVelocity();
        const double *robot_ang = robot->getOrientation();

        VectorXd x(n_x);
        x << robot_pos[0], robot_pos[1], robot_pos[2], robot_vel[0], robot_vel[1], robot_vel[2],
             robot_ang[0], robot_ang[1], robot_ang[2], robot_vel[3], robot_vel[4], robot_vel[5];

        VectorXd x_ref = VectorXd::Zero(N * n_x);
        for (int i = 0; i < N; ++i) {
            x_ref.segment(i * n_x, n_x) << 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        }

        VectorXd force = modelPredictiveControl(x, x_ref);

        double force_A[3] = {force(0), force(1), force(2)};
        double offset_A[3] = {0.222, 0.2, 0};

        double force_B[3] = {force(3), force(4), force(5)};
        double offset_B[3] = {0.222, -0.2, 0};
        
        double force_C[3] = {force(6), force(7), force(8)};
        double offset_C[3] = {-0.222, -0.2, 0};
        
        double force_D[3] = {force(9), force(10), force(11)};
        double offset_D[3] = {-0.222, 0.2, 0};

        robot->addForceWithOffset(force_A, offset_A, false);
        robot->addForceWithOffset(force_B, offset_B, false);
        robot->addForceWithOffset(force_C, offset_C, false);
        robot->addForceWithOffset(force_D, offset_D, false);

        std::cout << "- - -" << std::endl;
        std::cout << "Robot Pos: [" << robot_pos[0] << ", " << robot_pos[1] << ", " << robot_pos[2] << "]" << std::endl;
        std::cout << "Robot Vel: [" << robot_vel[0] << ", " << robot_vel[1] << ", " << robot_vel[2] << "]" << std::endl;
        std::cout << "Force A: [" << force(0) << ", " << force(1)  << ", " << force(2)  << "]" << std::endl;
        std::cout << "Force B: [" << force(3) << ", " << force(4)  << ", " << force(5)  << "]" << std::endl;
        std::cout << "Force C: [" << force(6) << ", " << force(7)  << ", " << force(8)  << "]" << std::endl;
        std::cout << "Force D: [" << force(9) << ", " << force(10) << ", " << force(11) << "]" << std::endl;
        std::cout << "= = = = =" << std::endl;

        loop_count++;
    }
    return 0;
}
