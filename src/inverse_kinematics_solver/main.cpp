#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for handling std::array
#include "franka_ik_He.hpp"

namespace py = pybind11;

PYBIND11_MODULE(franka_ik, m) {
    m.doc() = "Inverse kinematics solver"; // optional module docstring

    m.def("ik_solver", &franka_IK_EE, "A function that computes inverse kinematics for Panda emika robot");
}