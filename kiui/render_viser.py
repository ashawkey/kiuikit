import os
import tyro
import time
import trimesh
import numpy as np

try:
    import viser
    import viser.transforms as tf
except Exception as e:
    print('[WARN] try to install viser with `pip install viser`')
    os.system('pip install viser')
    import viser
    import viser.transforms as tf

from dataclasses import dataclass

@dataclass
class Options:
    # mesh path
    mesh: tyro.conf.Positional[str]
    # server port
    port: int = 8080


class ViserGUI:
    def __init__(self, opt: Options):
        self.opt = opt
        self.server = viser.ViserServer(port=opt.port)

        # load mesh
        self.wireframe = False
        self.mesh = self.load_mesh()

        # bg color
        self.bg_image = np.ones((1, 1, 3), dtype=np.uint8) * 255
        self.server.set_background_image(self.bg_image)

        self.register_gui()
    
    def load_mesh(self):
        mesh = trimesh.load(self.opt.mesh, force='mesh')
        # auto-resize
        vmin, vmax = np.split(mesh.bounding_box.bounds, 2)
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 2 * 0.8 / np.max(vmax - vmin)
        mesh.vertices = (mesh.vertices - self.ori_center) * self.ori_scale
        
        if self.wireframe:
            return self.server.add_mesh_simple(
                '/mesh', mesh.vertices, mesh.faces, 
                wxyz=tf.SO3.from_x_radians(np.pi/2).wxyz,
                position=(0, 0, 0),
                wireframe=self.wireframe,
            )
        else:
            return self.server.add_mesh_trimesh(
                '/mesh', mesh, 
                wxyz=tf.SO3.from_x_radians(np.pi/2).wxyz,
                position=(0, 0, 0),
            )
    
    def register_gui(self):
        
        with self.server.add_gui_folder("Render"):
            # mesh center position
            mesh_position = self.server.add_gui_vector3("Position", initial_value=(0.0, 0.0, 0.0), step=0.1)
            @mesh_position.on_update
            def _(_):
                self.mesh.position = mesh_position.value

            # wireframe mode
            mesh_wireframe = self.server.add_gui_checkbox("Wireframe", initial_value=False)
            @mesh_wireframe.on_update
            def _(_):
                self.wireframe = mesh_wireframe.value
                self.load_mesh()
            
            # bg color
            bg_color = self.server.add_gui_rgb("Background Color", initial_value=(255, 255, 255))
            @bg_color.on_update
            def _(_):
                self.bg_image = np.ones((1, 1, 3), dtype=np.uint8) * np.array(bg_color.value)
                self.server.set_background_image(self.bg_image)
            
    
    def render(self):
        while True:
            time.sleep(1)


def main():    
    opt = tyro.cli(Options)
    gui = ViserGUI(opt)
    gui.render()

if __name__ == "__main__":
    main()
