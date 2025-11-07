"""keyboard_kinematics controller."""
from controller import Robot, Motor, Keyboard
import math

MAX_SPEED = 10;
BASE_LENGTH = 0.1;

# create the Robot instance.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

wheel1 = robot.getDevice('wheel1')
wheel2 = robot.getDevice('wheel2')
wheel3 = robot.getDevice('wheel3')

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


KEY_VELOCITY_MAP = {
    Keyboard.LEFT: ('psi', MAX_SPEED), 
    Keyboard.RIGHT: ('psi', -MAX_SPEED),
    ord('W'): ('y', MAX_SPEED),
    ord('S'): ('y', -MAX_SPEED),
    ord('A'): ('x', MAX_SPEED),
    ord('D'): ('x', -MAX_SPEED),
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
    
def handle_keyboard():
    x_vel = 0.0
    y_vel = 0.0
    psi_vel = 0.0
    
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
            
    return (x_vel, y_vel, psi_vel)
            
# --- Main Loop ---
while robot.step(timestep) != -1:

    vels = handle_keyboard()
    x_vel = vels[0]
    y_vel = vels[1]
    psi_vel = vels[2]
    
    v1, v2, v3 = get_ik(x_vel, y_vel, 10 * psi_vel)
    
    wheel1.setVelocity(v1)
    wheel2.setVelocity(v2)
    wheel3.setVelocity(v3)

# Enter here exit cleanup code.