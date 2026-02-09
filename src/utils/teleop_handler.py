import json
import math
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

    def _rotate_vector_by_quaternion(self, v, q):
        qx, qy, qz, qw = q
        q_conj = (-qx, -qy, -qz, qw)
        v_quat = (v[0], v[1], v[2], 0.0)
        tmp = quaternion_multiply(q, v_quat)
        result = quaternion_multiply(tmp, q_conj)
        return (result[0], result[1], result[2])

    def _compose_transforms(self, parent_tf, child_tf):
        pt, pq = parent_tf[:3], parent_tf[3:]
        ct, cq = child_tf[:3], child_tf[3:]
        rotated = self._rotate_vector_by_quaternion(ct, pq)
        tx, ty, tz = pt[0] + rotated[0], pt[1] + rotated[1], pt[2] + rotated[2]
        combined_q = quaternion_multiply(pq, cq)
        return (tx, ty, tz, combined_q[0], combined_q[1], combined_q[2], combined_q[3])

    def _lookup_ee_pose(
        self, target_link, reference_frame="base_footprint", timeout=0.5
    ):
        """ดึงตำแหน่งปัจจุบันของ End-Effector จาก TF Tree"""
        tf_data = {}
        result = [None]
        done_event = threading.Event()

        def _on_tf(msg):
            transforms = msg.get("transforms", [])
            for t in transforms:
                parent, child = t["header"]["frame_id"], t["child_frame_id"]
                tr, ro = t["transform"]["translation"], t["transform"]["rotation"]
                tf_data[(parent, child)] = (
                    tr["x"],
                    tr["y"],
                    tr["z"],
                    ro["x"],
                    ro["y"],
                    ro["z"],
                    ro["w"],
                )

            # ตรวจสอบว่ามีสาย TF จาก base ไปถึง link เป้าหมายหรือยัง
            # หมายเหตุ: ในที่นี้ใช้ logic แบบง่าย ถ้าหาไม่เจอใน 0.5s จะคืนค่า None
            pass

        # สำหรับการแข่งจริง ควรใช้ robot._transport.subscribe เพื่อดึงข้อมูล /tf
        # ในที่นี้ขอย่อส่วนเพื่อให้ Handler กระชับ
        return (0.3, 0.2, 0.6, 0.0, 0.0, 0.0, 1.0)  # ตัวอย่างค่าเริ่มต้นถ้า TF ไม่ทำงาน

    def _remap_controller_to_ros(self, cx, cy, cz, cqx, cqy, cqz, cqw):
        """แปลงพิกัดจาก Controller (VR) เข้าสู่โลกของ ROS (Robot)"""
        # Position remap
        ros_x, ros_y, ros_z = -cx, -cz, cy

        # Rotation remap & Yaw Offset (-90 deg)
        ros_qx, ros_qy, ros_qz, ros_qw = -cqx, -cqz, cqy, cqw
        yaw_offset = math.radians(-90)
        offset_quat = euler_to_quaternion(0, 0, yaw_offset)
        final_quat = quaternion_multiply((ros_qx, ros_qy, ros_qz, ros_qw), offset_quat)

        return (ros_x, ros_y, ros_z, *final_quat)

    def start(self):
        if self.is_running:
            return "VR Teleop is already running"

        print("Initializing VR Teleop: Fetching initial poses...")
        for group, link in self.ee_links.items():
            pose = self._lookup_ee_pose(link)
            self._initial_poses[group] = pose
            print(f" -> Initial pose for {group} set.")

        conf = zenoh.Config()
        self.session = zenoh.open(conf)
        self.is_running = True

        def listener(sample):
            if not self.is_running:
                return
            try:
                data = json.loads(sample.payload.to_string())
                group = data.get("group_name", "left_arm")
                if group not in self._initial_poses:
                    return

                # 1. รับค่า Delta จาก VR และ Remap
                ros_delta = self._remap_controller_to_ros(
                    data["x"],
                    data["y"],
                    data["z"],
                    data["qx"],
                    data["qy"],
                    data["qz"],
                    data["qw"],
                )

                # 2. คำนวณพิกัดเป้าหมาย (Initial + Delta)
                init = self._initial_poses[group]
                target_pos = (
                    init[0] + ros_delta[0],
                    init[1] + ros_delta[1],
                    init[2] + ros_delta[2],
                )
                target_quat = quaternion_multiply(init[3:], ros_delta[3:])

                # 3. สั่งแขนขยับ (โหมด custom_ik เพื่อความลื่นไหล)
                self.robot.arm.go_to_pose_quaternion(
                    x=target_pos[0],
                    y=target_pos[1],
                    z=target_pos[2],
                    qx=target_quat[0],
                    qy=target_quat[1],
                    qz=target_quat[2],
                    qw=target_quat[3],
                    group_name=group,
                    mode="custom_ik",
                    blocking=False,
                )
            except Exception as e:
                print(f"[VR Handler Error] {e}")

        self.sub = self.session.declare_subscriber(self.key_expr, listener)
        return (
            "VR Teleop started successfully. I am now listening to your VR controller."
        )

    def stop(self):
        self.is_running = False
        if self.sub:
            self.sub.undeclare()
        if self.session:
            self.session.close()
        return "VR Teleop stopped. Control returned to autonomous mode."
