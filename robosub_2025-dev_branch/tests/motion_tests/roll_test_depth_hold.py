import rospy
import time

from auv.motion import robot_control
from auv.utils import arm, disarm


def main():
    rospy.init_node("roll_depth_hold_refined", anonymous=True)
    rc = robot_control.RobotControl(enable_dvl=False)

    arm.arm()
    time.sleep(3.0)

    rc.set_depth(0.7)
    rc.set_mode("ALT_HOLD")
    rospy.sleep(5.0)

    
    first_time = time.time()
    while time.time() - first_time < 5:
        rc.movement(roll=5)
    

    rc.movement(roll=0)
    time.sleep(1.0)

    disarm.disarm()


if __name__ == "__main__":
    main()
