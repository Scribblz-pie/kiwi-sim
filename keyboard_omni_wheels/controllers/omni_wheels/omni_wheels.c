/*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description: Controller for a three-omni-wheels (Kiwi) robot.
 * Implements Inverse Kinematics based on the user-provided matrix definition.
 * Control is handled via WASD (linear) and QE (angular) keyboard input.
 */

#include <stdio.h>
#include <math.h>
#include <webots/motor.h>
#include <webots/robot.h>
#include <webots/keyboard.h>

// --- CONSTANTS AND DEFINITIONS ---

// Webots time step (ms)
#define TIME_STEP 8

// Speed factor (rad/s). This is the maximum desired wheel angular velocity.
#define MAX_SPEED 8.0

// Scaling factor for angular velocity (L in the user's matrix, or L/r).
// Adjust this to change the robot's rotation sensitivity.
#define L_FACTOR 0.1

// Kinematic constants
#define SQRT3_OVER_2 0.86602540378
#define HALF_ONE 0.5

// Wheel device tags
static WbDeviceTag wheels[3];
static const char *wheel_names[3] = {"wheel1", "wheel2", "wheel3"};

/**
 * @brief Initialize all required Webots components.
 */
static void initialize_robot() {
  // Initialize Webots
  wb_robot_init();

  // Initialize motors
  for (int i = 0; i < 3; i++) {
    wheels[i] = wb_robot_get_device(wheel_names[i]);
    // Set motor to velocity control mode
    wb_motor_set_position(wheels[i], INFINITY);
    // Ensure motors start still
    wb_motor_set_velocity(wheels[i], 0.0);
  }

  // Enable keyboard input
  wb_keyboard_enable(TIME_STEP);
}

/**
 * @brief Calculate and set motor velocities using Inverse Kinematics.
 *
 * The desired angular velocities (omega) for the wheels are derived from the
 * desired robot body velocities (linear_x, linear_y, angular_phi):
 *
 * w1 = 0*x_dot + 1*y_dot + L*phi_dot
 * w2 = -0.866*x_dot - 0.5*y_dot + L*phi_dot
 * w3 = 0.866*x_dot - 0.5*y_dot + L*phi_dot
 *
 * @param linear_x Desired velocity in the body X-direction (strafe).
 * @param linear_y Desired velocity in the body Y-direction (forward/backward).
 * @param angular_phi Desired angular velocity (rotation).
 */
static void apply_inverse_kinematics(double linear_x, double linear_y, double angular_phi) {
  double w1, w2, w3;

  // Wheel 1 velocity
  w1 = linear_y + L_FACTOR * angular_phi;

  // Wheel 2 velocity
  w2 = -SQRT3_OVER_2 * linear_x - HALF_ONE * linear_y + L_FACTOR * angular_phi;

  // Wheel 3 velocity
  w3 = SQRT3_OVER_2 * linear_x - HALF_ONE * linear_y + L_FACTOR * angular_phi;

  // Apply the calculated velocities (which are in rad/s)
  wb_motor_set_velocity(wheels[0], w1);
  wb_motor_set_velocity(wheels[1], w2);
  wb_motor_set_velocity(wheels[2], w3);
}

/**
 * @brief Main control loop. Handles keyboard input and controls the robot.
 */
int main() {
  initialize_robot();

  int key;
  double v_x_dot = 0.0;     // Desired body velocity (strafe)
  double v_y_dot = 0.0;     // Desired body velocity (forward/backward)
  double v_phi_dot = 0.0;   // Desired angular velocity (rotation)

  // Webots main loop
  while (wb_robot_step(TIME_STEP) != -1) {

    // Reset desired velocities in each step
    v_x_dot = 0.0;
    v_y_dot = 0.0;
    v_phi_dot = 0.0;

    // Get the last key pressed
    key = wb_keyboard_get_key();

    // Handle keyboard input (W, A, S, D for linear, Q, E for angular)
    switch (key) {
      // Linear Motion (Forward/Backward - Y-axis)
      case 'W': // W key
      case WB_KEYBOARD_UP:
        v_y_dot = MAX_SPEED;
        break;
      case 'S': // S key
      case WB_KEYBOARD_DOWN:
        v_y_dot = -MAX_SPEED;
        break;

      // Linear Motion (Strafe Left/Right - X-axis)
      case 'A': // A key
      case WB_KEYBOARD_LEFT:
        v_x_dot = -MAX_SPEED;
        break;
      case 'D': // D key
      case WB_KEYBOARD_RIGHT:
        v_x_dot = MAX_SPEED;
        break;

      // Angular Motion (Rotation - Phi-axis)
      case 'Q': // Q key (Rotate Left)
        v_phi_dot = -MAX_SPEED;
        break;
      case 'E': // E key (Rotate Right)
        v_phi_dot = MAX_SPEED;
        break;

      default:
        // No key pressed or unrecognized key, all velocities remain 0.0
        break;
    }

    // Calculate and set wheel velocities
    apply_inverse_kinematics(v_x_dot, v_y_dot, v_phi_dot);
  }

  return 0;
}