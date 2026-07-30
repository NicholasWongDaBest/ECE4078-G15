# for computing the wheel calibration parameters
import os
import sys
import time
import argparse
import numpy as np
sys.path.insert(0, "..")
from botconnect import BotConnect # access the robot communication


def calibrateScale():
    # The scale parameter is used to convert the raw speed specified by you (0 to 1) to actual speed.
    # That is, actual speed = raw speed * scale
    # Actual speed is m/s, whereas raw speed unit is M/s where M is assumed arbitrary unit
    # We can get actual speed measurements by driving the robot at a fixed raw speed for a known distance (eg 1 meter) and record the time taken for it.
    # Repeat the procedures multiple times (can use different raw speed), to obtain the average value for a more robust measurement.

    # Feel free to change the range and do more repetition
    wheel_speed_range = [[0.4, 0.4], [0.45, 0.45], [0.5, 0.5]]
    delta_times = []

    for wheel_speed in wheel_speed_range:
        print("Driving at {} M/s.".format(wheel_speed))
        
        # Repeat the test until the correct time is found.      
        while True:
            delta_time = float(input("Input the time to drive in seconds: "))
            botconnect.move_auto_time(wheel_speed, delta_time)
            uInput = input("Did the robot travel 1m? [y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot drove 1m in {:.2f} seconds at wheel speed {}.\n".format(delta_time, wheel_speed))
                break

    # Once finished driving, compute the scale parameter using wheel_speed and delta_time. Remember to take the average.
    # Helpful tips: the unit of the scale parameter is m/M.
    num = len(wheel_speed_range)
    scale = 0
    for delta_time, wheel_speed in zip(delta_times, wheel_speed_range):
        pass # TODO: compute the scale parameter
    print("The scale parameter is estimated as {:.6f} m/M.".format(scale))

    return scale


def calibrateBaseline(scale):
    # The baseline parameter is the distance between the wheels.
    # This part is similar to the calibrateScale function, difference is that the robot is spinning 360 degree.
    # From the wheel_speed and delta_time, find out mathematically how to calculate the baseline.

    # Feel free to change the range / step and do more repetition
    wheel_speed_range = [[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]
    delta_times = []

    for wheel_speed in wheel_speed_range:
        print("Driving at {} M/s.".format(wheel_speed))
        
        # Repeat the test until the correct time is found.      
        while True:
            delta_time = float(input("Input the time to drive in seconds: "))
            botconnect.move_auto_time(wheel_speed, delta_time)
            uInput = input("Did the robot spin 360 degree? [y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot spun 360 degree in {:.2f} seconds at wheel speed {}.\n".format(delta_time, wheel_speed))
                break

    # Once finished driving, compute the baseline parameter using wheel_speed and delta_time. Remember to take the average.
    # Helpful tips: the unit of the baseline parameter is m. Think about the circumference of a circle. You may also need the scale parameter here.
    num = len(wheel_speed_range)
    baseline = 0
    for delta_time, wheel_speed in zip(delta_times, wheel_speed_range):
        pass # TODO: replace with your code to compute the baseline parameter using scale, wheel_speed, and delta_time
    print("The baseline parameter is estimated as {:.6f} m.".format(baseline))

    return baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    args, _ = parser.parse_known_args()
    
    botconnect = BotConnect(args.ip)
    botconnect.set_pid(use_pid=1, kp=???, ki=???, kd=???) # TODO: replace with your best constants

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd())

    print('Calibrating PiBot scale...\n')
    scale = calibrateScale()
    fileNameS = "{}scale.txt".format(dataDir)
    np.savetxt(fileNameS, np.array([scale]), delimiter=',')

    print('Calibrating PiBot baseline...\n')
    baseline = calibrateBaseline(scale)
    fileNameB = "{}baseline.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline]), delimiter=',')

    print('Finished wheel calibration')