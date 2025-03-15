#include <webots/Supervisor.hpp>
#include <webots/Node.hpp>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "qpOASES.hpp"

using namespace webots;

std::vector<std::vector<double>> compute_forces(double m, double g) {
    std::vector<std::vector<double>> forces(4, std::vector<double>(3, 0.0));

    double force_per_leg = m * g / 4.0;
    for (int i = 0; i < 4; i++) {
        forces[i][2] = force_per_leg;
    }

    return forces;
}

int main(int argc, char **argv) {
    Supervisor *supervisor = new Supervisor();
    Node *robot = supervisor->getFromDef("robot");

    int timestep = supervisor->getBasicTimeStep();

    double m = 22;
    double g = 9.81;

    const double *robot_pos = robot->getPosition();
    const double *robot_ang = robot->getOrientation();

    while (supervisor->step(timestep) != -1) {
        // Update robot state.
        robot_pos = robot->getPosition();
        robot_ang = robot->getOrientation();

        std::cout << "Robot Pos = " << std::endl
                  << "[" << robot_pos[0] << ", " << robot_pos[1] << ", " << robot_pos[2] << "]" << std::endl
                  << "- - -" << std::endl;

        std::cout << "Robot Ang = " << std::endl
                  << "[" << robot_ang[0] << ", " << robot_ang[1] << ", " << robot_ang[2] << std::endl
                  << " " << robot_ang[3] << ", " << robot_ang[4] << ", " << robot_ang[5] << std::endl
                  << " " << robot_ang[6] << ", " << robot_ang[7] << ", " << robot_ang[8] << "]" << std::endl
                  << "- - -" << std::endl;

        // Compute the optimal hip forces.
        std::vector<std::vector<double>> forces = compute_forces(m, g);

        // Offsets remain zero in this example.
        std::vector<std::vector<double>> offsets(4, std::vector<double>(3, 0.0));

        // Apply the computed forces to the robot.
        for (int i = 0; i < 4; i++) {
            robot->addForceWithOffset(forces[i].data(), offsets[i].data(), true);
        }

        std::cout << "= = = = =" << std::endl;
    }

    delete supervisor;
    return 0;
}
