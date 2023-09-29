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
        self.wogui = opt.wogui # disable gui and run in cmd
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32).cuda() # default white bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5
        
        self.mode = opt.mode
        self.render_modes = ['albedo', 'depth', 'normal', 'lambertian']

        # load mesh
        self.mesh = Mesh.load(opt.mesh, front_dir=opt.front_dir)

        if not opt.force_cuda_rast and (self.wogui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        if not self.wogui:
            dpg.create_context()
            self.register_dpg()
            self.step()
        

    def __del__(self):
        if not self.wogui:
            dpg.destroy_context()
    
    def step(self):

        if self.need_update:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            # do MVP for vertices
            pose = torch.from_numpy(self.cam.pose.astype(np.float32)).cuda()
            proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
            
            v_cam = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
            v_clip = v_cam @ proj.T

            rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (self.H, self.W))

            alpha = (rast[..., 3:] > 0).float()
            alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
            
            if self.mode == 'depth':
                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
            else:
                # use vertex color if exists
                if self.mesh.vc is not None:
                    albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
                # use texture image
                else:
                    texc, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft)
                    albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]

                albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
                albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).clamp(0, 1) # [1, H, W, 3]
                if self.mode == 'albedo':
                    albedo = albedo * alpha + self.bg_color * (1 - alpha)
                    buffer = albedo[0].detach().cpu().numpy()
                else:
                    normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                    normal = safe_normalize(normal)
                    if self.mode == 'normal':
                        normal_image = (normal[0] + 1) / 2
                        normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
                        buffer = normal_image.detach().cpu().numpy()
                    elif self.mode == 'lambertian':
                        light_d = np.deg2rad(self.light_dir)
                        light_d = np.array([
                            np.sin(light_d[0]) * np.sin(light_d[1]),
                            np.cos(light_d[0]),
                            np.sin(light_d[0]) * np.cos(light_d[1]),
                        ], dtype=np.float32)
                        light_d = torch.from_numpy(light_d).to(albedo.device)
                        lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
                        albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                        buffer = albedo[0].detach().cpu().numpy()


            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if self.need_update:
                self.render_buffer = buffer
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + buffer) / (self.spp + 1)

            if not self.wogui:
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
        with dpg.window(label="Control", tag="_control_window", width=300, height=200, collapsed=True):

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
                
                dpg.add_combo(self.render_modes, label='mode', default_value=self.mode, tag="_mode_combo", callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32).cuda() # only need RGB in [0, 1]
                    self.need_update = True
                
                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

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


        ### register IO handlers

        # camera mouse controller
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
        
        # press spacebar to toggle rendering mode
        def callback_space_toggle_mode(sender, app_data):
            self.mode = self.render_modes[(self.render_modes.index(self.mode) + 1) % len(self.render_modes)]
            dpg.set_value("_mode_combo", self.mode)
            self.need_update = True


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=callback_space_toggle_mode)

        
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
        assert not self.wogui
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':
    import os
    import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str, help="path to mesh (obj, ply, glb, ...)")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument("--wogui", action='store_true', help="disable all dpg GUI")
    parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
    parser.add_argument('--save', type=str, default=None, help="path to save example rendered images")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")
    parser.add_argument('--save_video', type=str, default=None, help="path to save rendered video")

    opt = parser.parse_args()

    gui = GUI(opt)

    if opt.save is not None:
        os.makedirs(opt.save, exist_ok=True)
        # render from fixed views and save all images
        elevation = [opt.elevation,]
        azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
        for ele in tqdm.tqdm(elevation):
            for azi in tqdm.tqdm(azimuth):
                gui.cam.from_angle(ele, azi)
                gui.need_update = True
                gui.step()
                if not opt.wogui:
                    dpg.render_dearpygui_frame()
                image = (gui.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(opt.save, f'{ele}_{azi}.png'), image)
    elif opt.save_video is not None:
        import imageio
        images = []
        elevation = [opt.elevation,]
        # azimuth = np.arange(-180, 180, 2, dtype=np.int32) # back-->front-->back
        azimuth = np.arange(0, 360, 2, dtype=np.int32) # front-->back-->front
        for ele in tqdm.tqdm(elevation):
            for azi in tqdm.tqdm(azimuth):
                gui.cam.from_angle(ele, azi)
                gui.need_update = True
                gui.step()
                if not opt.wogui:
                    dpg.render_dearpygui_frame()
                image = (gui.render_buffer * 255).astype(np.uint8)
                images.append(image)
        images = np.stack(images, axis=0)
        # ~6 seconds, 180 frames at 30 fps
        imageio.mimwrite(opt.save_video, images, fps=30, quality=8, macro_block_size=1)
    else:
        gui.render()