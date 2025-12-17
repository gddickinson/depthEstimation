import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl
import bpy
import mathutils

class FBXSkeleton:
    def __init__(self, fbx_file_path):
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import FBX
        bpy.ops.import_scene.fbx(filepath=fbx_file_path)

        self.armature = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
        if not self.armature:
            raise ValueError("No armature found in the FBX file")

        self.joints = {}
        self.meshes = []
        self.bones = []
        self.create_skeleton_structure()
        self.create_skeleton_mesh()

    def create_skeleton_structure(self):
        def traverse_bones(bone, parent=None):
            joint = SkeletonJoint(bone.name, parent)
            self.joints[bone.name] = joint
            for child in bone.children:
                traverse_bones(child, joint)

        for bone in self.armature.data.bones:
            if bone.parent is None:  # Root bone
                traverse_bones(bone)

    def create_skeleton_mesh(self):
        for joint in self.joints.values():
            if joint.parent:
                bone = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]), color=(1,1,1,1), width=2, antialias=True)
                self.bones.append((joint.name, bone))

            mesh = gl.MeshData.sphere(rows=10, cols=20, radius=0.02)
            joint_item = gl.GLMeshItem(meshdata=mesh, smooth=True, color=(1,0,0,1))
            self.meshes.append((joint.name, joint_item))

    def update_joint_positions(self, frame):
        bpy.context.scene.frame_set(frame)
        for joint_name, joint in self.joints.items():
            pose_bone = self.armature.pose.bones[joint_name]
            global_pos = self.armature.matrix_world @ pose_bone.matrix @ mathutils.Vector((0, 0, 0))
            joint.position = np.array([global_pos.x, global_pos.y, global_pos.z])

    def update_mesh(self):
        for joint_name, bone in self.bones:
            joint = self.joints[joint_name]
            if joint.parent:
                start_pos = joint.parent.position
                end_pos = joint.position
                bone.setData(pos=np.array([start_pos, end_pos]))

        for joint_name, mesh in self.meshes:
            joint_pos = self.joints[joint_name].position
            mesh.resetTransform()
            mesh.translate(*joint_pos)

class SkeletonJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.position = np.zeros(3)

        if parent:
            parent.children.append(self)

class FBXSkeletonTestWindow(QMainWindow):
    def __init__(self, fbx_file_path):
        super().__init__()
        self.setWindowTitle("FBX Skeleton Animation Test")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)

        self.skeleton = FBXSkeleton(fbx_file_path)
        for _, bone in self.skeleton.bones:
            self.view.addItem(bone)
        for _, mesh in self.skeleton.meshes:
            self.view.addItem(mesh)

        self.view.setCameraPosition(distance=5, elevation=30, azimuth=45)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_skeleton)
        self.timer.start(50)  # Update every 50 ms

        self.frame = 0

    def animate_skeleton(self):
        self.skeleton.update_joint_positions(self.frame)
        self.skeleton.update_mesh()
        self.frame += 1
        if self.frame >= bpy.context.scene.frame_end:
            self.frame = bpy.context.scene.frame_start

def main():
    app = QApplication(sys.argv)
    fbx_file_path = '/Users/george/Desktop/fbx files/DT_Puppet_Rig.blend'# Replace with your FBX file path
    window = FBXSkeletonTestWindow(fbx_file_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
