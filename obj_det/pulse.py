import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
left = 3
right = 5

GPIO.setup(left, GPIO.OUT)
GPIO.setup(right, GPIO.OUT)

while True:
    GPIO.output(right, GPIO.HIGH)
    print("High")
    time.sleep(2)
    GPIO.output(right, GPIO.LOW)
    print("Low")
    time.sleep(5)
