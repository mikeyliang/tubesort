import RPi.GPIO as GPIO
import time



def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(X_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(Y_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(Z_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def read_switches():
    x_state = not GPIO.input(X_LIMIT_PIN) # Active LOW due to pull-up resistor
    y_state = not GPIO.input(Y_LIMIT_PIN)
    z_state = not GPIO.input(Z_LIMIT_PIN)


    print(x_state)
    return x_state, y_state, z_state


while True:
    read_switches()