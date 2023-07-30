import cv2
import numpy as np

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

import dearpygui.dearpygui as dpg

from kiui.mesh import Mesh
from kiui.cam import OrbitCamera
from kiui.op import safe_normalize


class GUI:
    def __init__(self, opt, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5
        
        self.mode = 'lambertian'

        # load mesh
        self.mesh = Mesh.load(opt.mesh)

        self.glctx = dr.RasterizeCudaContext() # dr.RasterizeGLContext()

        dpg.create_context()
        self.register_dpg()
        self.step()
        

    def __del__(self):
        dpg.destroy_context()
    
    def step(self):

        if self.need_update:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            # do MVP for vertices
            mv = torch.from_numpy(self.cam.view).cuda() # [4, 4]
            proj = torch.from_numpy(self.cam.perspective).cuda() # [4, 4]
            mvp = proj @ mv
            
            v_clip = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0),
                              torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]

            rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (self.H, self.W))
            
            if self.mode == 'depth':
                depth = rast[0, :, :, [2]]  # [H, W, 1]
                buffer = depth.detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
            else:
                texc, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft)
                albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
                albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
                albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).clamp(0, 1) # [1, H, W, 3]
                if self.mode == 'albedo':
                    buffer = albedo[0].detach().cpu().numpy()
                else:
                    normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                    normal = safe_normalize(normal)
                    if self.mode == 'normal':
                        buffer = (normal[0].detach().cpu().numpy() + 1) / 2
                    elif self.mode == 'lambertian':
                        light_d = np.deg2rad(self.light_dir)
                        light_d = np.array([
                            np.sin(light_d[0]) * np.sin(light_d[1]),
                            np.cos(light_d[0]),
                            np.sin(light_d[0]) * np.cos(light_d[1]),
                        ], dtype=np.float32)
                        light_d = torch.from_numpy(light_d).to(albedo.device)
                        lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
                        buffer = (albedo * lambertian.unsqueeze(-1))[0].detach().cpu().numpy()


            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if self.need_update:
                self.render_buffer = buffer
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + buffer) / (self.spp + 1)

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=200):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)              

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('albedo', 'depth', 'normal', 'lambertian'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(label="FoVY", min_value=1, max_value=120, format="%d deg", default_value=np.rad2deg(self.cam.fovy), callback=callback_set_fovy)

                # light dir
                def callback_set_light_dir(sender, app_data, user_data):
                    self.light_dir[user_data] = app_data
                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Plane Light Direction:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="theta", min_value=0, max_value=180, format="%.2f", default_value=self.light_dir[0], callback=callback_set_light_dir, user_data=0)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="phi", min_value=0, max_value=360, format="%.2f", default_value=self.light_dir[1], callback=callback_set_light_dir, user_data=1)

                # ambient ratio
                def callback_set_abm_ratio(sender, app_data):
                    self.ambient_ratio = app_data
                    self.need_update = True

                dpg.add_slider_float(label="ambient", min_value=0, max_value=1.0, format="%.5f", default_value=self.ambient_ratio, callback=callback_set_abm_ratio)


            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='mesh viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':
    import os
    import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', default='', type=str, help="path to mesh (obj, glb, ...)")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=30, help="default GUI camera fovy")
    parser.add_argument('--save', type=str, default=None, help="path to save example rendered images")

    opt = parser.parse_args()

    gui = GUI(opt)

    if opt.save is not None:
        os.makedirs(opt.save, exist_ok=True)
        # render from fixed views and save all images
        elevation = [0, -30, -60]
        azimuth = np.arange(-180, 180, 10, dtype=np.int32)
        for ele in tqdm.tqdm(elevation):
            for azi in tqdm.tqdm(azimuth):
                gui.cam.from_angle(ele, azi)
                gui.need_update = True
                gui.step()
                dpg.render_dearpygui_frame()
                image = (gui.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(opt.save, f'{ele}_{azi}.png'), image)
    else:
        gui.render()