import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl
import bpy
import bmesh
import math

class FBXAnimator:
    def __init__(self, fbx_file_path):
        print("Initializing FBXAnimator...")

        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import FBX
        bpy.ops.import_scene.fbx(filepath=fbx_file_path)

        self.armature = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
        if not self.armature:
            raise ValueError("No armature found in the FBX file")

        self.mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        if not self.mesh_objects:
            raise ValueError("No mesh found in the FBX file")

        print(f"Armature: {self.armature.name}")
        print(f"Mesh objects: {[obj.name for obj in self.mesh_objects]}")

        self.verify_armature_modifier()
        self.verify_vertex_groups()
        self.vertices = []
        self.faces = []
        self.create_mesh_data()
        self.setup_animation()
        self.prev_vertices = None

    def verify_armature_modifier(self):
        for obj in self.mesh_objects:
            armature_modifier = next((mod for mod in obj.modifiers if mod.type == 'ARMATURE'), None)
            if not armature_modifier:
                print(f"Adding Armature modifier to {obj.name}")
                armature_modifier = obj.modifiers.new(name="Armature", type='ARMATURE')

            if armature_modifier.object != self.armature:
                print(f"Setting Armature modifier object for {obj.name}")
                armature_modifier.object = self.armature

            # Move Armature modifier to the top of the stack
            while obj.modifiers[0] != armature_modifier:
                bpy.ops.object.modifier_move_up({"object": obj}, modifier=armature_modifier.name)

            print(f"Armature modifier verified for {obj.name}")

    def verify_vertex_groups(self):
        for obj in self.mesh_objects:
            print(f"Verifying vertex groups for {obj.name}")
            mesh_vertex_groups = set(vg.name for vg in obj.vertex_groups)
            armature_bones = set(bone.name for bone in self.armature.data.bones)

            missing_groups = armature_bones - mesh_vertex_groups
            if missing_groups:
                print(f"  Missing vertex groups: {missing_groups}")
            else:
                print("  All necessary vertex groups are present")

            # Check if vertices are assigned to groups
            vertices_in_groups = sum(1 for v in obj.data.vertices if v.groups)
            print(f"  Vertices assigned to groups: {vertices_in_groups}/{len(obj.data.vertices)}")

    def create_mesh_data(self):
        print("Creating mesh data...")
        all_vertices = []
        all_faces = []
        offset = 0
        for obj in self.mesh_objects:
            # Create a bmesh
            bm = bmesh.new()
            bm.from_mesh(obj.data)

            # Triangulate faces
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

            # Update the bmesh
            bm.to_mesh(obj.data)
            bm.free()

            # Now get the triangulated mesh data
            mesh = obj.data
            vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
            faces = np.array([[v for v in f.vertices] for f in mesh.polygons])
            faces += offset

            all_vertices.extend(vertices)
            all_faces.extend(faces)
            offset += len(vertices)

        self.vertices = np.array(all_vertices)
        self.faces = np.array(all_faces)

        # Center and scale the mesh
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        scale = 2.0 / np.max(np.abs(self.vertices))
        self.vertices *= scale

        print(f"Mesh data created. Vertices: {len(self.vertices)}, Faces: {len(self.faces)}")

    def setup_animation(self):
        print("Setting up animation...")
        if not self.armature.animation_data:
            self.armature.animation_data_create()
        action = bpy.data.actions.new(name="SimpleAnimation")
        self.armature.animation_data.action = action

        # Animate some key bones
        bones_to_animate = ['spine_03', 'upperarm_l', 'upperarm_r', 'lowerleg_l', 'lowerleg_r']
        for bone_name in bones_to_animate:
            bone = self.armature.pose.bones.get(bone_name)
            if bone:
                for i in range(3):  # x, y, z rotation
                    fc = action.fcurves.new(data_path=f'pose.bones["{bone_name}"].rotation_euler', index=i)
                    fc.keyframe_points.add(2)
                    fc.keyframe_points[0].co = 0, 0
                    fc.keyframe_points[1].co = 100, math.pi/2  # 90 degree rotation over 100 frames

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 100
        print(f"Animation set up. Frame range: {bpy.context.scene.frame_start} to {bpy.context.scene.frame_end}")

    def update_mesh(self, frame):
        print(f"Updating mesh for frame {frame}")
        bpy.context.scene.frame_set(frame)

        # Force update of the armature
        self.armature.update_tag()
        bpy.context.view_layer.update()

        # Debug: Print some bone transformations
        bones_to_print = ['spine_03', 'upperarm_l', 'upperarm_r', 'lowerleg_l', 'lowerleg_r']
        for bone_name in bones_to_print:
            bone = self.armature.pose.bones.get(bone_name)
            if bone:
                print(f"Bone {bone_name}: Rotation = {bone.rotation_euler}")

        # Get the updated vertex positions
        depsgraph = bpy.context.evaluated_depsgraph_get()
        all_vertices = []
        for obj in self.mesh_objects:
            obj_eval = obj.evaluated_get(depsgraph)

            # Attempt to manually apply the armature modifier
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply({"object": obj}, modifier="Armature")

            mesh = obj_eval.to_mesh()

            try:
                vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
                all_vertices.extend(vertices)
            except Exception as e:
                print(f"Error processing vertices: {str(e)}")

            obj_eval.to_mesh_clear()

        self.vertices = np.array(all_vertices)

        # Center and scale the mesh
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        scale = 2.0 / np.max(np.abs(self.vertices))
        self.vertices *= scale

        print(f"Mesh updated. New vertex count: {len(self.vertices)}")
        print(f"Sample vertices: {self.vertices[:5]}")

        # Debug: Check for changes in vertex positions
        if self.prev_vertices is not None:
            vertex_diff = np.abs(self.vertices - self.prev_vertices).max()
            print(f"Maximum vertex position change: {vertex_diff}")
        self.prev_vertices = self.vertices.copy()

# The rest of the code (FBXViewerWindow and main function) remains the same


class FBXViewerWindow(QMainWindow):
    def __init__(self, fbx_file_path):
        super().__init__()
        self.setWindowTitle("FBX Animation Viewer")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)

        self.fbx_animator = FBXAnimator(fbx_file_path)

        self.mesh_item = gl.GLMeshItem(vertexes=self.fbx_animator.vertices, faces=self.fbx_animator.faces,
                                       smooth=True, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.view.addItem(self.mesh_item)

        self.view.setCameraPosition(distance=5, elevation=30, azimuth=45)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_mesh)
        self.timer.start(50)  # Update every 50 ms (20 fps)

        self.frame = bpy.context.scene.frame_start

    def animate_mesh(self):
        print(f"\nAnimating mesh, frame: {self.frame}")
        self.fbx_animator.update_mesh(self.frame)
        self.mesh_item.setMeshData(vertexes=self.fbx_animator.vertices, faces=self.fbx_animator.faces)

        self.view.update()  # Force update of the view

        self.frame += 1
        if self.frame > bpy.context.scene.frame_end:
            self.frame = bpy.context.scene.frame_start

def main():
    app = QApplication(sys.argv)
    fbx_file_path = '/Users/george/Desktop/fbx files/renderpeople_free_rigged_people_FBX/rp_eric_rigged_001_FBX/rp_eric_rigged_001_zup_a.fbx'  # Replace with your FBX file path
    window = FBXViewerWindow(fbx_file_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
