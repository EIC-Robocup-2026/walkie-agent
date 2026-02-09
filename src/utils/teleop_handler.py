import json
import threading
import time

import zenoh
from walkie_sdk import WalkieRobot
from walkie_sdk.utils.converters import euler_to_quaternion, quaternion_multiply


class VRTeleopHandler:
    def __init__(self, robot: WalkieRobot, key_expr: str = "arm_pose"):
        self.robot = robot
        self.key_expr = key_expr
        self.session = None
        self.sub = None
        self.is_running = False
        self._initial_poses = {}
        self.ee_links = {"left_arm": "left_link7", "right_arm": "right_link7"}

    def _remap_logic(self, data):
        # นำ logic จาก remap_controller_to_ros ใน examples มาใส่
        pass

    def start(self):
        if self.is_running:
            return "VR Teleop is already running"

        # 1. (Initial Pose) via TF
        for group, link in self.ee_links.items():
            # ใช้ logic lookup_ee_pose จากไฟล์ตัวอย่าง
            self._initial_poses[group] = (
                0.2,
                -0.3,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
            )  # ค่า Dummy

        # 2. Open Zenoh Session
        conf = zenoh.Config()
        self.session = zenoh.open(conf)
        self.is_running = True

        # 3. Create Subscriber
        def listener(sample):
            if not self.is_running:
                return
            # รับข้อมูล ทำ Remap และสั่ง robot.arm.go_to_pose_quaternion(..., mode='custom_ik')
            pass

        self.sub = self.session.declare_subscriber(self.key_expr, listener)
        return "VR Teleop started successfully"

    def stop(self):
        self.is_running = False
        if self.sub:
            self.sub.undeclare()
        if self.session:
            self.session.close()
        return "VR Teleop stopped"
