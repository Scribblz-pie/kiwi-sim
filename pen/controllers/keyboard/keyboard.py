from controller import Robot, Motor, Keyboard
import math

MAX_SPEED = 10
BASE_LENGTH = 0.1

# create the Robot instance.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

wheel1 = robot.getDevice('wheel1')
wheel2 = robot.getDevice('wheel2')
wheel3 = robot.getDevice('wheel3')
pen = robot.getDevice('pen')

# Set motors to velocity control mode by setting position to infinity
wheel1.setPosition(float('inf'))
wheel2.setPosition(float('inf'))
wheel3.setPosition(float('inf'))

# Initialize motor velocities to zero
wheel1.setVelocity(0.0)
wheel2.setVelocity(0.0)
wheel3.setVelocity(0.0)

# Get and enable keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

pen_on = False
pen.write(pen_on)

KEY_VELOCITY_MAP = {
    Keyboard.LEFT: ('psi', -MAX_SPEED),
    Keyboard.RIGHT: ('psi', MAX_SPEED),
    ord('W'): ('x', -MAX_SPEED),
    ord('S'): ('x', MAX_SPEED),
    ord('A'): ('y', -MAX_SPEED),
    ord('D'): ('y', MAX_SPEED),
}

def get_ik(x_vel, y_vel, psi_vel):
    """Returns a tubel of the motor velocities from the robot velocities (inverse kinematics)"""
    l = BASE_LENGTH
    x = x_vel
    y = y_vel
    psi = psi_vel
    v1 = y + l * psi
    v2 = -(math.sqrt(3) * (3 * x + math.sqrt(3) * y - 2 * math.sqrt(3) * l * psi)) / 6
    v3 = ((math.sqrt(3) * x) / 2) - (y/2) + l * psi

    return (v1, v2, v3)

def handle_keyboard(current_pen_state):
    x_vel = 0.0
    y_vel = 0.0
    psi_vel = 0.0
    new_pen_state = current_pen_state

    keys_pressed = set()
    key = keyboard.getKey()

    while key != -1:
        keys_pressed.add(key)
        key = keyboard.getKey()

    for key_code in keys_pressed:
        if key_code in KEY_VELOCITY_MAP:
            axis, value = KEY_VELOCITY_MAP[key_code]

            if axis == 'psi':
                psi_vel = value
            elif axis == 'y':
                y_vel = value
            elif axis == 'x':
                x_vel = value
        elif key_code == ord('Y') or key_code == ord('X'):
            new_pen_state = not current_pen_state

    return (x_vel, y_vel, psi_vel, new_pen_state)

# --- Main Loop ---
while robot.step(timestep) != -1:

    vels_and_pen = handle_keyboard(pen_on)
    x_vel = vels_and_pen[0]
    y_vel = vels_and_pen[1]
    psi_vel = vels_and_pen[2]
    pen_on = vels_and_pen[3]

    v1, v2, v3 = get_ik(x_vel, y_vel, 10 * psi_vel)

    wheel1.setVelocity(v1)
    wheel2.setVelocity(v2)
    wheel3.setVelocity(v3)
    pen.write(pen_on)