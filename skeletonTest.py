import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl


class SkeletonJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.position = np.zeros(3)
        self.initial_offset = np.zeros(3)

    def add_child(self, child):
        self.children.append(child)

class Skeleton:
    def __init__(self):
        self.joints = {}
        self.bones = []
        self.meshes = []
        self.create_skeleton_structure()
        self.create_skeleton_mesh()

    def create_skeleton_structure(self):
        # Create joints
        joints = [
            'pelvis', 'spine_naval', 'spine_chest', 'neck', 'clavicle_left', 'shoulder_left', 'elbow_left', 'wrist_left',
            'hand_left', 'handtip_left', 'thumb_left', 'clavicle_right', 'shoulder_right', 'elbow_right', 'wrist_right',
            'hand_right', 'handtip_right', 'thumb_right', 'hip_left', 'knee_left', 'ankle_left', 'foot_left',
            'hip_right', 'knee_right', 'ankle_right', 'foot_right', 'head', 'nose', 'eye_left', 'ear_left',
            'eye_right', 'ear_right'
        ]

        for joint in joints:
            self.joints[joint] = SkeletonJoint(joint)

        # Define hierarchy
        hierarchy = [
            ('pelvis', ['spine_naval', 'hip_left', 'hip_right']),
            ('spine_naval', ['spine_chest']),
            ('spine_chest', ['neck', 'clavicle_left', 'clavicle_right']),
            ('neck', ['head']),
            ('head', ['nose', 'eye_left', 'eye_right', 'ear_left', 'ear_right']),
            ('clavicle_left', ['shoulder_left']),
            ('shoulder_left', ['elbow_left']),
            ('elbow_left', ['wrist_left']),
            ('wrist_left', ['hand_left']),
            ('hand_left', ['handtip_left', 'thumb_left']),
            ('clavicle_right', ['shoulder_right']),
            ('shoulder_right', ['elbow_right']),
            ('elbow_right', ['wrist_right']),
            ('wrist_right', ['hand_right']),
            ('hand_right', ['handtip_right', 'thumb_right']),
            ('hip_left', ['knee_left']),
            ('knee_left', ['ankle_left']),
            ('ankle_left', ['foot_left']),
            ('hip_right', ['knee_right']),
            ('knee_right', ['ankle_right']),
            ('ankle_right', ['foot_right'])
        ]

        for parent, children in hierarchy:
            for child in children:
                self.joints[parent].add_child(self.joints[child])
                self.joints[child].parent = self.joints[parent]

        # Set initial offsets (you may need to adjust these values)
        self.set_initial_offsets()

    def set_initial_offsets(self):
        # These are example offset values, adjust as needed for your model
        self.joints['spine_naval'].initial_offset = np.array([0, 0, 0.2])
        self.joints['spine_chest'].initial_offset = np.array([0, 0, 0.2])
        self.joints['neck'].initial_offset = np.array([0, 0, 0.2])
        self.joints['head'].initial_offset = np.array([0, 0, 0.1])
        self.joints['clavicle_left'].initial_offset = np.array([-0.1, 0, 0.1])
        self.joints['clavicle_right'].initial_offset = np.array([0.1, 0, 0.1])
        self.joints['shoulder_left'].initial_offset = np.array([-0.2, 0, 0])
        self.joints['shoulder_right'].initial_offset = np.array([0.2, 0, 0])
        self.joints['elbow_left'].initial_offset = np.array([-0.3, 0, 0])
        self.joints['elbow_right'].initial_offset = np.array([0.3, 0, 0])
        self.joints['wrist_left'].initial_offset = np.array([-0.25, 0, 0])
        self.joints['wrist_right'].initial_offset = np.array([0.25, 0, 0])
        self.joints['hand_left'].initial_offset = np.array([-0.1, 0, 0])
        self.joints['hand_right'].initial_offset = np.array([0.1, 0, 0])
        self.joints['handtip_left'].initial_offset = np.array([-0.05, 0, 0])
        self.joints['handtip_right'].initial_offset = np.array([0.05, 0, 0])
        self.joints['thumb_left'].initial_offset = np.array([-0.03, 0.03, 0])
        self.joints['thumb_right'].initial_offset = np.array([0.03, 0.03, 0])
        self.joints['hip_left'].initial_offset = np.array([-0.1, 0, -0.1])
        self.joints['hip_right'].initial_offset = np.array([0.1, 0, -0.1])
        self.joints['knee_left'].initial_offset = np.array([0, 0, -0.4])
        self.joints['knee_right'].initial_offset = np.array([0, 0, -0.4])
        self.joints['ankle_left'].initial_offset = np.array([0, 0, -0.4])
        self.joints['ankle_right'].initial_offset = np.array([0, 0, -0.4])
        self.joints['foot_left'].initial_offset = np.array([0, 0.1, 0])
        self.joints['foot_right'].initial_offset = np.array([0, 0.1, 0])
        self.joints['nose'].initial_offset = np.array([0, 0.1, 0])
        self.joints['eye_left'].initial_offset = np.array([-0.03, 0.07, 0.04])
        self.joints['eye_right'].initial_offset = np.array([0.03, 0.07, 0.04])
        self.joints['ear_left'].initial_offset = np.array([-0.07, 0, 0.03])
        self.joints['ear_right'].initial_offset = np.array([0.07, 0, 0.03])

    def create_skeleton_mesh(self):
        # Create bones
        for joint in self.joints.values():
            if joint.parent:
                bone = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]), color=(1,1,1,1), width=2, antialias=True)
                self.bones.append((joint.name, bone))

        # Create joint spheres
        for joint in self.joints.values():
            mesh = gl.MeshData.sphere(rows=10, cols=20, radius=0.02)
            joint_item = gl.GLMeshItem(meshdata=mesh, smooth=True, color=(1,0,0,1))
            self.meshes.append((joint.name, joint_item))

        # Create larger spheres for hands and feet
        for joint_name in ['hand_left', 'hand_right', 'foot_left', 'foot_right']:
            mesh = gl.MeshData.sphere(rows=10, cols=20, radius=0.04)
            joint_item = gl.GLMeshItem(meshdata=mesh, smooth=True, color=(0,1,0,1))
            self.meshes.append((f"{joint_name}_large", joint_item))

        # Create head
        head_mesh = gl.MeshData.sphere(rows=10, cols=20, radius=0.1)
        self.head = gl.GLMeshItem(meshdata=head_mesh, smooth=True, color=(1,1,0,1))
        self.meshes.append(('head_large', self.head))

    def update_joint_positions(self, joint_data):
        for joint_name, position in joint_data.items():
            if joint_name in self.joints:
                self.joints[joint_name].position = np.array(position)

        # Update positions of joints without direct data
        self.update_dependent_joints()

    def update_dependent_joints(self):
        for joint in self.joints.values():
            if joint.parent and np.all(joint.position == 0):
                parent_pos = joint.parent.position
                joint.position = parent_pos + joint.initial_offset

    def update_mesh(self):
        for joint_name, bone in self.bones:
            joint = self.joints[joint_name]
            if joint.parent:
                start_pos = joint.parent.position
                end_pos = joint.position
                bone.setData(pos=np.array([start_pos, end_pos]))

        for joint_name, mesh in self.meshes:
            if joint_name.endswith('_large'):
                joint_name = joint_name[:-6]  # Remove '_large' suffix
            joint_pos = self.joints[joint_name].position
            mesh.resetTransform()
            mesh.translate(*joint_pos)

class SkeletonTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Skeleton Test")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)

        self.skeleton = Skeleton()
        for _, bone in self.skeleton.bones:
            self.view.addItem(bone)
        for _, mesh in self.skeleton.meshes:
            self.view.addItem(mesh)

        # Set up camera
        self.view.setCameraPosition(distance=3, elevation=30, azimuth=45)

        # Set up animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_skeleton)
        self.timer.start(50)  # Update every 50 ms

        self.frame = 0

    def animate_skeleton(self):
        # Example animation - replace this with actual Kinect data in your main application
        t = self.frame * 0.1
        joint_data = {
            'pelvis': [0, 0, np.sin(t) * 0.1],
            'spine_naval': [0, 0, 0.2 + np.sin(t) * 0.05],
            'spine_chest': [0, 0, 0.4 + np.sin(t) * 0.05],
            'neck': [0, 0, 0.6 + np.sin(t) * 0.02],
            'head': [np.sin(t*0.7) * 0.05, 0, 0.7 + np.sin(t) * 0.02],
            'shoulder_left': [-0.2, 0, 0.5 + np.sin(t+1) * 0.05],
            'elbow_left': [-0.4, np.sin(t) * 0.1, 0.3 + np.cos(t) * 0.1],
            'wrist_left': [-0.6, np.sin(t) * 0.15, 0.2 + np.cos(t) * 0.15],
            'shoulder_right': [0.2, 0, 0.5 + np.sin(t) * 0.05],
            'elbow_right': [0.4, np.sin(t+2) * 0.1, 0.3 + np.cos(t+2) * 0.1],
            'wrist_right': [0.6, np.sin(t+2) * 0.15, 0.2 + np.cos(t+2) * 0.15],
            'hip_left': [-0.1, 0, -0.1 + np.sin(t+1) * 0.05],
            'knee_left': [-0.15, 0, -0.5 + np.sin(t+2) * 0.1],
            'ankle_left': [-0.1, 0.1, -0.9 + np.sin(t+3) * 0.05],
            'hip_right': [0.1, 0, -0.1 + np.sin(t+2) * 0.05],
            'knee_right': [0.15, 0, -0.5 + np.sin(t+3) * 0.1],
            'ankle_right': [0.1, 0.1, -0.9 + np.sin(t+4) * 0.05],
        }

        self.skeleton.update_joint_positions(joint_data)
        self.skeleton.update_mesh()
        self.frame += 1

def main():
    app = QApplication(sys.argv)
    window = SkeletonTestWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
