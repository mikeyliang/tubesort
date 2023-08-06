

import RPi.GPIO as GPIO
import time
import datetime
import numpy as np
import threading


class Motor(object):
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        self.x_pins = [4, 5, 6, 17]
        self.y_pins = [23, 24, 25, 26]
        self.z_pins = [12, 16, 20, 21]
        # Set pin numbers
        self.limit_switch_x_pin = 20
        self.limit_switch_y_pin = 21

        GPIO.setup(self.limit_switch_x_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.limit_switch_y_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.x_unit_step = 200
        self.y_unit_step = 200
        self.z_unit_step = 26
        self.current_pos = [None, None, None]
        self.origin_coordinate = [None, None, None]

        for pin in self.x_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        for pin in self.y_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        self.HALF_STEP = [[1, 0, 0, 0],
                          [1, 1, 0, 0],
                          [0, 1, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 1],
                          [0, 0, 0, 1],
                          [1, 0, 0, 1],]

    def __del__(self):
        GPIO.cleanup()

    def calibrate(self):
        print("Calibrating...")

        # Move the motors until the limit switches are triggered
        # Here we are moving in the negative direction as an example
        while not GPIO.input(self.limit_switch_x_pin):
            self.move('x', -1)

        print('x limit switch triggered')

        while not GPIO.input(self.limit_switch_y_pin):
            self.move('y', -1)

        print('y limit switch triggered')

        # Once the limit switches are triggered, we set the current coordinates as the origin
        self.origin_coordinate = [0, 0, self.current_pos[2]]
        self.current_pos = [0, 0, self.current_pos[2]]

        print("Calibration complete.")

    def moveToPoints()

    def move_by_coordinate(self, x_position, y_position):
        # x = 11inch, y = 10inch, z = 3 inch
        dx_block = x_position-self.current_pos[0]
        dy_block = y_position-self.current_corrdinate[1]
        x_direction = -1 if dx_block > 0 else 1
        y_direction = 1 if dy_block > 0 else -1

        if is_multiThread is False:
            self.move('x', x_direction * abs(dx_block))
            time.sleep(0.5)
            self.move('y', y_direction * abs(dy_block))
            time.sleep(0.5)
        else:
            self.move_xy([x_direction * abs(dx_block),
                         y_direction * abs(dy_block)], True)

    def move(self, axis, blocks):
        if axis == 'x':
            pins = self.x_pins
            step = round(abs(blocks) * self.x_unit_step)

        elif axis == 'y':
            pins = self.y_pins
            step = round(abs(blocks) * self.y_unit_step)

        else:
            pins = self.z_pins
            step = round(abs(blocks) * self.z_unit_step)

        for _ in range(step):
            if (blocks > 0):
                for step in self.HALF_STEP:
                    for pin, value in zip(pins, step):
                        GPIO.output(pin, value)
                    time.sleep(0.001)
            else:
                for step in self.HALF_STEP:
                    for pin, value in zip(pins, step):
                        GPIO.output(pin, value)
                    time.sleep(0.001)

    def move_xy(self, distance, is_multiThread=False):
        if is_multiThread is True:
            thread_y = threading.Thread(
                target=self.move, args=('y', distance[1]))
            thread_y.start()

            self.move('x', distance[0])

        else:
            self.move('x', distance[0])
            time.sleep(0.5)
            self.move('y', distance[1])
            time.sleep(0.5)
