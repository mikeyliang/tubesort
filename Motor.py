

import RPi.GPIO as GPIO
import time
import datetime
import numpy as np
import threading

# Setup GPIO pins
# Setup GPIO pins
X_DIR = 20   # Direction pin
X_STEP = 21  # Step pin
Y_DIR = 27
Y_STEP = 28
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation

# Pin Definitions
X_LIMIT_PIN = 9
Y_LIMIT_PIN = 10
Z_LIMIT_PIN = 11

# Step the motor
step_count = 200  # Number of steps to take
delay = 0.005  # Time delay between steps. Adjust as needed for your motor/driver combination.


    

# Directional control for  motors
def move_motor(motor_pin, direction):
    if direction == 'CW':
        GPIO.output(motor_pin, GPIO.HIGH)
    elif direction == 'CCW':
        GPIO.output(motor_pin, GPIO.LOW)
    else:
        print("Unknown direction!")


class Motor(object):
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        
        # setup x motor pins
        GPIO.setup(X_DIR, GPIO.OUT)
        GPIO.setup(X_STEP, GPIO.OUT)
        GPIO.output(X_DIR, CW)
        
        # setup y motor pins
        GPIO.setup(Y_DIR, GPIO.OUT)
        GPIO.setup(Y_STEP, GPIO.OUT)
        GPIO.output(Y_DIR, CW)

        # setup limit switch pins
        GPIO.setup(X_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(Y_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(Z_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.current_pos = [0, 0, 0]

    def __del__(self):
        GPIO.cleanup()


    def read_switches():
        x_state = not GPIO.input(X_LIMIT_PIN) # Active LOW due to pull-up resistor
        y_state = not GPIO.input(Y_LIMIT_PIN)
        z_state = not GPIO.input(Z_LIMIT_PIN)
        return x_state, y_state, z_state

    def calibrate(self):
        print("Calibrating...")

        # Wait for the x limit switch to be triggered
        while GPIO.input(X_LIMIT_PIN):  # This is the correct condition
            self.move('x', -1)
        print('x limit switch triggered')
        
        # Reset the current X position after hitting the limit
        self.current_pos[0] = 0

        # Wait for the y limit switch to be triggered
        while GPIO.input(Y_LIMIT_PIN):  # This is the correct condition
            self.move('y', -1)
        print('y limit switch triggered')
        
        # Reset the current Y position after hitting the limit
        self.current_pos[1] = 0

        print("Calibration complete.")


    def move_by_coordinate(self, x_position, y_position, is_multiThread=False):
        # x = 11inch, y = 10inch, z = 3 inch
        dx_block = x_position - self.current_pos[0]
        dy_block = y_position - self.current_pos[1]
        
        x_direction = 1 if dx_block > 0 else -1
        y_direction = 1 if dy_block > 0 else -1

        if not is_multiThread:
            self.move('x', x_direction * abs(dx_block))
            time.sleep(0.5)
            self.move('y', y_direction * abs(dy_block))
            time.sleep(0.5)
        else:
            self.move_xy([x_direction * abs(dx_block), y_direction * abs(dy_block)], True)

    def move(self, axis, blocks: int):
        # Determine direction and step pins based on the axis
        if axis == 'x':
            dir_pin = X_DIR
            step_pin = X_STEP
            position_index = 0
        elif axis == 'y':
            dir_pin = Y_DIR
            step_pin = Y_STEP
            position_index = 1
        else:
            print("Unknown axis!")
            return False

        # Set the direction
        direction = GPIO.HIGH if blocks > 0 else GPIO.LOW
        GPIO.output(dir_pin, direction)

        # Move the stepper motor
        steps = abs(blocks)
        for _ in range(steps):
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(delay)  # pulse duration
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(delay)  # delay between steps

        # Update current position
        self.current_pos[position_index] += blocks

    def move_xy(self, distance, is_multiThread=False):
        if is_multiThread:
            thread_y = threading.Thread(target=self.move, args=('y', distance[1]))
            thread_y.start()
            self.move('x', distance[0])
        else:
            self.move('x', distance[0])
            time.sleep(0.5)
            self.move('y', distance[1])
            time.sleep(0.5)





try:
    motor = Motor()
    motor.calibrate()
finally:
    GPIO.cleanup()

