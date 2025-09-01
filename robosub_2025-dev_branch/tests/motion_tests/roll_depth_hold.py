#!/usr/bin/env python3
"""Simple test script to perform a roll while keeping a fixed depth.

The script uses the :mod:`auv.motion.robot_control` helper to command the
Pixhawk. Depth hold is activated by switching to ``ALT_HOLD`` mode before
starting the roll. The vertical thrusters are then managed by the existing
PID controller while a roll command is repeatedly sent in ``ACRO`` mode.
"""

import time

import rospy
from auv.motion import robot_control
from auv.utils import arm, disarm


def main():
    rospy.init_node("roll_depth_hold_test", anonymous=True)

    rc = robot_control.RobotControl(enable_dvl=False)

    arm.arm()
    rc.set_mode("ALT_HOLD")
    rc.set_depth(0.5)  # Target depth in metres
    time.sleep(5.0)

    rc.set_mode("ACRO")

    roll_power = 2
    roll_duration = 5.0
    start = time.time()
    while time.time() - start < roll_duration and not rospy.is_shutdown():
        rc.movement(roll=roll_power)
        time.sleep(0.1)

    rc.movement(roll=0)
    rc.set_mode("STABILIZE")
    disarm.disarm()


if __name__ == "__main__":
    main()
