#include <webots/Supervisor.hpp>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <iostream>
#include <cmath>

using namespace webots;

// Constants
const int m = 22;
const double gravity = 9.81;
const double dt = 0.001;
const int N = 10;
const int n_x = 12;
const int n_u = 12;

// Global matrices for system dynamics and cost
Eigen::MatrixXd A(n_x, n_x);
Eigen::MatrixXd B(n_x, n_u);
Eigen::MatrixXd Q(n_x, n_x);
Eigen::MatrixXd R(n_u, n_u);

// Function to compute Euler angles (roll, pitch, yaw) from rotation matrix.
void getEulerAngles(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw) {
    // Compute pitch from the rotation matrix
    pitch = atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    
    // Handle potential singularities at pitch = ±pi/2
    if (fabs(pitch - M_PI/2) < 1e-6) {
        roll = 0;
        yaw = atan2(R(1, 2), R(0, 2));
    } else if (fabs(pitch + M_PI/2) < 1e-6) {
        roll = 0;
        yaw = -atan2(R(1, 2), R(0, 2));
    } else {
        roll = atan2(R(2, 1), R(2, 2));
        yaw = atan2(R(1, 0), R(0, 0));
    }
}

// Converts an array of 9 doubles into an Eigen 3x3 rotation matrix.
Eigen::Matrix3d getRotationMatrix(const double *robot_rot) {
    Eigen::Matrix3d R;
    R << robot_rot[0], robot_rot[1], robot_rot[2],
         robot_rot[3], robot_rot[4], robot_rot[5],
         robot_rot[6], robot_rot[7], robot_rot[8];
    return R;
}

// Initialize the system dynamics and cost matrices.
// The offsets for each actuator (or leg) are used to compute B.
void initializeMatrices(const double *ra, const double *rb, const double *rc, const double *rd) {
    // State transition matrix A (discrete-time integrator model)
    A << 1, 0, 0, dt, 0,0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, dt,0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, dt,0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 0, 0, dt,0, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt,0,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    // Control input matrix B.
    B << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         dt/m, 0, 0, dt/m, 0, 0, dt/m, 0, 0, dt/m, 0, 0,
         0, dt/m, 0, 0, dt/m, 0, 0, dt/m, 0, 0, dt/m, 0,
         0, 0, dt/m, 0, 0, dt/m, 0, 0, dt/m, 0, 0, dt/m,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, -60/m*ra[2]*dt, 60/m*ra[1]*dt, 0, -60/m*rb[2]*dt, 60/m*rb[1]*dt, 0, -60/m*rc[2]*dt, 60/m*rc[1]*dt, 0, -60/m*rd[2]*dt, 60/m*rd[1]*dt,
         30/m*ra[2]*dt, 0, -30/m*ra[0]*dt, 30/m*rb[2]*dt, 0, -30/m*rb[0]*dt, 30/m*rc[2]*dt, 0, -30/m*rc[0]*dt, 30/m*rd[2]*dt, 0, -30/m*rd[0]*dt,
         -300/13/m*ra[1]*dt, 300/13/m*ra[0]*dt, 0, -300/13/m*rb[1]*dt, 300/13/m*rb[0]*dt, 0, -300/13/m*rc[1]*dt, 300/13/m*rc[0]*dt, 0, -300/13/m*rd[1]*dt, 300/13/m*rd[0]*dt;

    // Set up state and control cost matrices.
    Q = Eigen::MatrixXd::Zero(n_x, n_x);
    Q.diagonal() << 1e9, 1e9, 1e9, 1e6, 1e6, 1e6, 1e9, 1e9, 1e9, 1e6, 1e6, 1e6;
    R = Eigen::MatrixXd::Identity(n_u, n_u);
}

// Model Predictive Control function that solves a quadratic program.
// It returns the first control input of the optimal control sequence.
Eigen::VectorXd modelPredictiveControl(const Eigen::VectorXd &x, const Eigen::VectorXd &x_ref) {
    // Prediction matrices for state evolution over the horizon.
    Eigen::MatrixXd A_qp = Eigen::MatrixXd::Zero(N * n_x, n_x);
    Eigen::MatrixXd B_qp = Eigen::MatrixXd::Zero(N * n_x, (N - 1) * n_u);

    // Build prediction matrices
    for (int i = 0; i < N; ++i) {
        Eigen::MatrixXd A_pow = Eigen::MatrixXd::Identity(n_x, n_x);
        for (int k = 0; k < (i + 1); ++k){
            A_pow *= A;
        }
        A_qp.block(i * n_x, 0, n_x, n_x) = A_pow;

        int max_j = (i < (N - 1)) ? (i + 1) : (N - 1);
        for (int j = 0; j < max_j; ++j) {
            Eigen::MatrixXd A_temp = Eigen::MatrixXd::Identity(n_x, n_x);
            for (int k = 0; k < (i - j); ++k){
                A_temp *= A;
            }
            B_qp.block(i * n_x, j * n_u, n_x, n_u) = A_temp * B;
        }
    }

    // Assemble block diagonal cost matrices for the state and control.
    Eigen::MatrixXd Q_N = Eigen::MatrixXd::Zero(N * n_x, N * n_x);
    for (int i = 0; i < N; ++i){
        Q_N.block(i * n_x, i * n_x, n_x, n_x) = Q;
    }

    Eigen::MatrixXd R_N = Eigen::MatrixXd::Zero((N - 1) * n_u, (N - 1) * n_u);
    for (int i = 0; i < (N - 1); ++i){
        R_N.block(i * n_u, i * n_u, n_u, n_u) = R;
    }

    // Set up the quadratic cost function: 1/2 * u'Hu + g'u.
    Eigen::MatrixXd H = 2 * (B_qp.transpose() * Q_N * B_qp + R_N);
    Eigen::VectorXd g = 2 * B_qp.transpose() * Q_N * (A_qp * x - x_ref);

    // Regularize H to ensure positive definiteness.
    H += 1e-6 * Eigen::MatrixXd::Identity(H.rows(), H.cols());

    // Set up the QP solver.
    OsqpEigen::Solver solver;
    const int n_vars = (N - 1) * n_u;
    solver.data()->setNumberOfVariables(n_vars);
    solver.data()->setNumberOfConstraints(n_vars);

    Eigen::SparseMatrix<double> H_sparse = H.sparseView();
    solver.data()->setHessianMatrix(H_sparse);
    solver.data()->setGradient(g);

    // Box constraints (identity matrix for linear constraints).
    Eigen::SparseMatrix<double> A_constraint(n_vars, n_vars);
    A_constraint.setIdentity();
    solver.data()->setLinearConstraintsMatrix(A_constraint);

    Eigen::VectorXd lb = Eigen::VectorXd::Constant(n_vars, -300.0);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(n_vars, 300.0);
    solver.data()->setLowerBound(lb);
    solver.data()->setUpperBound(ub);

    // Initialize and solve the QP problem.
    solver.initSolver();
    solver.solveProblem();
    
    // Return the first control input.
    Eigen::VectorXd u_opt = solver.getSolution();
    return u_opt.head(n_u);
}

int main() {
    // Create supervisor instance and get robot node.
    Supervisor supervisor;
    Node *robot = supervisor.getFromDef("robot");
    int timestep = static_cast<int>(supervisor.getBasicTimeStep());

    // Simulation loop.
    while (supervisor.step(timestep) != -1) {
        // Get current state information.
        const double *robot_pos = robot->getPosition();
        const double *robot_vel = robot->getVelocity();
        const double *robot_rot = robot->getOrientation();
    
        // Compute rotation matrices and Euler angles.
        Eigen::Matrix3d R_mat = getRotationMatrix(robot_rot);
        Eigen::Matrix3d R_T = R_mat.transpose();
        double roll, pitch, yaw;
        getEulerAngles(R_mat, roll, pitch, yaw);

        // Assemble state vector.
        Eigen::VectorXd x(n_x);
        x << robot_pos[0], robot_pos[1], robot_pos[2],
            robot_vel[0], robot_vel[1], robot_vel[2],
            roll,         pitch,        yaw,
            robot_vel[3], robot_vel[4], robot_vel[5];

        // Define reference state trajectory (hover at z = 0.3, rest zeros).
        Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(N * n_x);
        for (int i = 0; i < N; ++i) {
            x_ref.segment(i * n_x, n_x) << 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        }

        // Define force offsets for each actuator (legs).
        double offset_A[3] = { 0.222,  0.2, 0};
        double offset_B[3] = { 0.222, -0.2, 0};
        double offset_C[3] = {-0.222, -0.2, 0};
        double offset_D[3] = {-0.222,  0.2, 0};

        // Initialize dynamics matrices using the offsets.
        initializeMatrices(offset_A, offset_B, offset_C, offset_D);

        // Compute optimal force vector via MPC.
        Eigen::VectorXd force = modelPredictiveControl(x, x_ref);

        // Extract forces for each actuator.
        double force_A[3] = {force(0), force(1), force(2)};
        double force_B[3] = {force(3), force(4), force(5)};
        double force_C[3] = {force(6), force(7), force(8)};
        double force_D[3] = {force(9), force(10), force(11)};

        // Convert computed global forces to the robot's local frame.
        auto convertForceToLocal = [&](double *f_global) {
            Eigen::Vector3d f_global_vec(f_global[0], f_global[1], f_global[2]);
            Eigen::Vector3d f_local = R_T * f_global_vec;
            f_global[0] = f_local(0);
            f_global[1] = f_local(1);
            f_global[2] = f_local(2);
        };

        convertForceToLocal(force_A);
        convertForceToLocal(force_B);
        convertForceToLocal(force_C);
        convertForceToLocal(force_D);

        // Apply forces to the robot at the specified offsets.
        robot->addForceWithOffset(force_A, offset_A, false);
        robot->addForceWithOffset(force_B, offset_B, false);
        robot->addForceWithOffset(force_C, offset_C, false);
        robot->addForceWithOffset(force_D, offset_D, false);

        // Print robot state and applied forces for debugging.
        std::cout << "- - -" << std::endl;
        std::cout << "Robot Pos: [" << robot_pos[0] << ", " << robot_pos[1] << ", " << robot_pos[2] << "]" << std::endl;
        std::cout << "Robot Vel: [" << robot_vel[0] << ", " << robot_vel[1] << ", " << robot_vel[2] << "]" << std::endl;
        std::cout << "Robot Ang: [" << roll << ", " << pitch << ", " << yaw << "]" << std::endl;
        std::cout << "Robot Ang Vel: [" << robot_vel[3] << ", " << robot_vel[4] << ", " << robot_vel[5] << "]" << std::endl;
        std::cout << "Force A: [" << force(0) << ", " << force(1) << ", " << force(2) << "]" << std::endl;
        std::cout << "Force B: [" << force(3) << ", " << force(4) << ", " << force(5) << "]" << std::endl;
        std::cout << "Force C: [" << force(6) << ", " << force(7) << ", " << force(8) << "]" << std::endl;
        std::cout << "Force D: [" << force(9) << ", " << force(10) << ", " << force(11) << "]" << std::endl;
        std::cout << "= = = = =" << std::endl;
    }
    return 0;
}
