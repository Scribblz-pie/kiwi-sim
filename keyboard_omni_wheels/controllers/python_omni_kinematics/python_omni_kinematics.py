from controller import Robot, Motor, Keyboard
import math
import numpy as np # Using numpy for explicit matrix definition

# --- CONSTANTS AND DEFINITIONS ---

# Webots time step (ms)
TIME_STEP = 8

# Speed factor (rad/s). This is the maximum desired wheel angular velocity.
MAX_SPEED = 8.0

# Scaling factor for angular velocity (L or L/r).
# Radius of the robot base (from center to wheel contact point).
L_FACTOR = 1 # Example value for L (radius of robot)

# Kinematic constants (derived from the Inverse Kinematics Matrix J_inv)
SQRT3_OVER_2 = math.sqrt(3) / 2.0  # ~ 0.866
HALF_ONE = 0.5

# Wheel device names
wheel_names = ["wheel1", "wheel2", "wheel3"]

# Inverse Kinematics Matrix (J_inv) - Used to map [x_dot, y_dot, phi_dot] to [w1, w2, w3]
# The user-provided Forward Kinematics Matrix J was:
# [[ 0, -sqrt(3)/3,  sqrt(3)/3],
#  [ 2/3,   -1/3,       -1/3],
#  [ 1/3L,  1/3L,       1/3L]]
# The Inverse Kinematics Matrix J_inv (used for control) is:
# [[ 0,   1,    L],
#  [-sqrt(3)/2, -1/2, L],
#  [ sqrt(3)/2, -1/2, L]]
J_inv = np.array([
    [0.0, 1.0, L_FACTOR],
    [-SQRT3_OVER_2, -HALF_ONE, L_FACTOR],
    [SQRT3_OVER_2, -HALF_ONE, L_FACTOR]
])


def apply_inverse_kinematics(wheels, linear_x, linear_y, angular_phi):
    """
    @brief Calculate and set motor velocities using Inverse Kinematics.

    The desired angular velocities (omega) for the wheels are derived from the
    desired robot body velocities (linear_x, linear_y, angular_phi):

    w1 = 0*x_dot + 1*y_dot + L*phi_dot
    w2 = -0.866*x_dot - 0.5*y_dot + L*phi_dot
    w3 = 0.866*x_dot - 0.5*y_dot + L*phi_dot

    :param wheels: List of Webots Motor objects.
    :param linear_x: Desired velocity in the body X-direction (strafe).
    :param linear_y: Desired velocity in the body Y-direction (forward/backward).
    :param angular_phi: Desired angular velocity (rotation).
    """

    # Wheel 1 velocity (0 * linear_x + 1 * linear_y + L_FACTOR * angular_phi)
    w1 = linear_y + L_FACTOR * angular_phi

    # Wheel 2 velocity (-SQRT3_OVER_2 * linear_x - HALF_ONE * linear_y + L_FACTOR * angular_phi)
    w2 = -SQRT3_OVER_2 * linear_x - HALF_ONE * linear_y + L_FACTOR * angular_phi

    # Wheel 3 velocity (SQRT3_OVER_2 * linear_x - HALF_ONE * linear_y + L_FACTOR * angular_phi)
    w3 = SQRT3_OVER_2 * linear_x - HALF_ONE * linear_y + L_FACTOR * angular_phi

    # Normalize velocities to ensure they don't exceed MAX_SPEED (optional but good practice)
    max_w = max(abs(w1), abs(w2), abs(w3))
    if max_w > MAX_SPEED:
        scale_factor = MAX_SPEED / max_w
        w1 *= scale_factor
        w2 *= scale_factor
        w3 *= scale_factor

    # Apply the calculated velocities (which are in rad/s)
    wheels[0].setVelocity(w1)
    wheels[1].setVelocity(w2)
    wheels[2].setVelocity(w3)


def main():
    # create the Robot instance.
    robot = Robot()

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    # Get motor devices and set them to velocity control
    wheels = []
    for name in wheel_names:
        motor = robot.getDevice(name)
        # Set motor to velocity control mode (position is set to infinity)
        motor.setPosition(float('inf'))
        # Ensure motors start still
        motor.setVelocity(0.0)
        wheels.append(motor)

    # Enable keyboard input
    keyboard = robot.getKeyboard()
    # Use the robot's basic timestep for keyboard sampling
    keyboard.enable(timestep) 

    # Main loop:
    while robot.step(timestep) != -1:
        
        # Reset desired velocities in each step
        v_x_dot = 0.0     # Desired body velocity (strafe)
        v_y_dot = 0.0     # Desired body velocity (forward/backward)
        v_phi_dot = 0.0   # Desired angular velocity (rotation)

        # --- KEYBOARD INPUT PROCESSING FOR SIMULTANEOUS CONTROL ---
        
        # Process the keyboard buffer to handle multiple keys pressed in the same time step.
        # This allows for simultaneous translation and rotation.
        key = keyboard.getKey()
        while key != -1:
            
            # LINEAR Y CONTROL (Forward/Backward)
            if key == ord('W') or key == Keyboard.UP:
                v_y_dot = MAX_SPEED
            elif key == ord('S') or key == Keyboard.DOWN:
                v_y_dot = -MAX_SPEED

            # LINEAR X CONTROL (Strafe Left/Right)
            # Removed Keyboard.LEFT/RIGHT from strafe to avoid conflict with rotation
            elif key == ord('A'):
                v_x_dot = -MAX_SPEED
            elif key == ord('D'):
                v_x_dot = MAX_SPEED

            # ANGULAR CONTROL (Rotation) - NEW MAPPING
            elif key == Keyboard.LEFT: # Rotate Left
                v_phi_dot = -MAX_SPEED
            elif key == Keyboard.RIGHT: # Rotate Right
                v_phi_dot = MAX_SPEED
            
            # Read the next key in the buffer
            key = keyboard.getKey()

        # Calculate and set wheel velocities
        apply_inverse_kinematics(wheels, v_x_dot, v_y_dot, v_phi_dot)


if __name__ == "__main__":
    main()