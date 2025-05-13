import os
import cv2
import math
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation

from kiui.cam import OrbitCamera

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from kiui.op import dot, safe_normalize

PRESET = {
    '2head': {"nose": [0.0021412528585642576, 0.09663597494363785, 0.0658300444483757], "neck": [-0.0010681250132620335, -0.00773019902408123, 0.0070612248964607716], "right_shoulder": [-0.05952326953411102, -0.01074729859828949, -9.221061918651685e-05], "right_elbow": [-0.08530594408512115, -0.055266208946704865, -0.0002990829525515437], "right_wrist": [-0.10009504109621048, -0.10672871768474579, 0.04145783558487892], "left_shoulder": [0.05843103677034378, -0.00984001811593771, 0.006993727292865515], "left_elbow": [0.08632118254899979, -0.05415782332420349, 0.0014727215748280287], "left_wrist": [0.10457248985767365, -0.10312359780073166, 0.03887445852160454], "right_hip": [0.045034412294626236, -0.11504126340150833, 0.013827108778059483], "right_knee": [0.04918656870722771, -0.18811877071857452, 0.013533061370253563], "right_ankle": [0.04868278279900551, -0.26335108280181885, 0.00865345261991024], "left_hip": [-0.042053237557411194, -0.12201181799173355, 0.01380667183548212], "left_knee": [-0.04329617694020271, -0.1887422651052475, 0.017976703122258186], "left_ankle": [-0.043604593724012375, -0.2664816379547119, 0.009848599322140217], "right_eye": [-0.059466104954481125, 0.14732250571250916, 0.03459283709526062], "left_eye": [0.05259571596980095, 0.15527287125587463, 0.03957919031381607], "right_ear": [-0.0915362536907196, 0.12822526693344116, -0.0068031493574380875], "left_ear": [0.0850512906908989, 0.132415309548378, -0.0023928845766931772]},
    '2.5head': {"nose": [0.0017373580485582352, 0.10973469167947769, 0.0656559020280838], "neck": [0.0005578577402047813, 0.04446818679571152, 0.007351499516516924], "right_shoulder": [-0.06537579745054245, 0.04295105114579201, 0.0007656214293092489], "right_elbow": [-0.10073791444301605, -0.011567563749849796, 0.0011918626260012388], "right_wrist": [-0.13308240473270416, -0.08192941546440125, 0.041919875890016556], "left_shoulder": [0.06524424999952316, 0.044958289712667465, 0.006919004488736391], "left_elbow": [0.10230650007724762, -0.003759381826967001, 0.0006817418616265059], "left_wrist": [0.13201594352722168, -0.0703246220946312, 0.037093378603458405], "right_hip": [0.04595021530985832, -0.08364222943782806, 0.014006344601511955], "right_knee": [0.04918656870722771, -0.18811877071857452, 0.013533061370253563], "right_ankle": [0.04868278279900551, -0.26335108280181885, 0.00865345261991024], "left_hip": [-0.04423205181956291, -0.09601262211799622, 0.014173348434269428], "left_knee": [-0.04329617694020271, -0.1887422651052475, 0.017976703122258186], "left_ankle": [-0.043604593724012375, -0.2664816379547119, 0.009848599322140217], "right_eye": [-0.059466104954481125, 0.14732250571250916, 0.03459283709526062], "left_eye": [0.05259571596980095, 0.15527287125587463, 0.03957919031381607], "right_ear": [-0.0915362536907196, 0.12822526693344116, -0.0068031493574380875], "left_ear": [0.0850495845079422, 0.1380147635936737, -0.002471053972840309]},
    '3head': {"nose": [-0.009920584969222546, 0.12076142430305481, 0.05144921690225601], "neck": [-0.010807228274643421, 0.0514158271253109, 0.0013299279380589724], "right_shoulder": [-0.06006723269820213, 0.04696957394480705, 0.0002115728275384754], "right_elbow": [-0.09811610728502274, -0.005623028613626957, 0.0059844981878995895], "right_wrist": [-0.13300932943820953, -0.06305573135614395, 0.03299811854958534], "left_shoulder": [0.03998754173517227, 0.04971584677696228, 0.00508818868547678], "left_elbow": [0.08185750991106033, -0.0020600110292434692, -0.00042760284850373864], "left_wrist": [0.12347455322742462, -0.057821620255708694, 0.03114679642021656], "right_hip": [0.028485914692282677, -0.07230513542890549, 0.007468733470886946], "right_knee": [0.02956254966557026, -0.1690925806760788, 0.0041032107546925545], "right_ankle": [0.03433872014284134, -0.26075273752212524, 0.004083261359483004], "left_hip": [-0.045689668506383896, -0.07934730499982834, 0.007511031813919544], "left_knee": [-0.04399966821074486, -0.17574959993362427, 0.004589484538882971], "left_ankle": [-0.04372630640864372, -0.26631468534469604, 0.004584244918078184], "right_eye": [-0.050951384007930756, 0.14704833924770355, 0.030185498297214508], "left_eye": [0.030040200799703598, 0.14831678569316864, 0.03128870204091072], "right_ear": [-0.07488956302404404, 0.12157893925905228, -0.004280052147805691], "left_ear": [0.05265972018241882, 0.12078605592250824, -0.004687455017119646]},
    '4head': {"nose": [-0.003130262019112706, 0.16587696969509125, 0.05414091795682907], "neck": [-0.008572826161980629, 0.10935179889202118, -0.005226037930697203], "right_shoulder": [-0.0681774765253067, 0.10397181659936905, -0.006579247768968344], "right_elbow": [-0.11421658098697662, 0.04033476859331131, 0.0004059926141053438], "right_wrist": [-0.1564374417066574, -0.02915881760418415, 0.033092476427555084], "left_shoulder": [0.05288884416222572, 0.10729481279850006, -0.0006785409059375525], "left_elbow": [0.10355149209499359, 0.0446460098028183, -0.007352650165557861], "left_wrist": [0.1539081186056137, -0.022825559601187706, 0.030852381139993668], "right_hip": [0.03897187486290932, -0.040350597351789474, 0.0022019187454134226], "right_knee": [0.04027460888028145, -0.15746350586414337, -0.001870364649221301], "right_ankle": [0.04605376720428467, -0.2683720886707306, -0.001894504064694047], "left_hip": [-0.05078059807419777, -0.04887162148952484, 0.002253100508823991], "left_knee": [-0.04873568192124367, -0.16551849246025085, -0.0012819726252928376], "left_ankle": [-0.04840493202209473, -0.27510207891464233, -0.0012883121380582452], "right_eye": [-0.03098677098751068, 0.19395537674427032, 0.019874906167387962], "left_eye": [0.01657041721045971, 0.1956009715795517, 0.02724142000079155], "right_ear": [-0.05411602929234505, 0.1733667254447937, -0.013280442915856838], "left_ear": [0.0373358279466629, 0.16922003030776978, -0.009465649724006653]},
    '7head': {"nose": [0.008811305277049541, 0.31194087862968445, 0.03809100389480591], "neck": [0.002824489725753665, 0.2497633546590805, -0.027212638407945633], "right_shoulder": [-0.06274063885211945, 0.2438453733921051, -0.0287011731415987], "right_elbow": [-0.11721517890691757, 0.11645109206438065, -0.020040860399603844], "right_wrist": [-0.14608919620513916, -0.027798010036349297, 0.013604634441435337], "left_shoulder": [0.07043224573135376, 0.24750067293643951, -0.02221038192510605], "left_elbow": [0.13446204364299774, 0.11769299954175949, -0.02934686467051506], "left_wrist": [0.1729350984096527, -0.029831381514668465, 0.013683688826858997], "right_hip": [0.05391363054513931, -0.017539208754897118, -0.0190418753772974], "right_knee": [0.06667664647102356, -0.1525234431028366, -0.023521387949585915], "right_ankle": [0.07457379996776581, -0.3374432921409607, -0.02354794181883335], "left_hip": [-0.059334054589271545, -0.023282308131456375, -0.018985575065016747], "left_knee": [-0.06742465496063232, -0.14818398654460907, -0.02287415601313114], "left_ankle": [-0.08158088475465775, -0.33846616744995117, -0.02288113348186016], "right_eye": [-0.0218308437615633, 0.34282705187797546, 0.0003984148206654936], "left_eye": [0.03048212267458439, 0.3446371853351593, 0.008501569740474224], "right_ear": [-0.04727301374077797, 0.3201795518398285, -0.0360724963247776], "left_ear": [0.05332399904727936, 0.31561821699142456, -0.03187622129917145]},
    '8head': {"nose": [0.013792905025184155, 0.3043023347854614, 0.031688809394836426], "neck": [0.009342477656900883, 0.2587159276008606, -0.022332727909088135], "right_shoulder": [-0.05664093792438507, 0.24332502484321594, -0.02351135015487671], "right_elbow": [-0.09274981170892715, 0.12393242120742798, -0.014478711411356926], "right_wrist": [-0.12370569258928299, -0.0065268343314528465, 0.014483341947197914], "left_shoulder": [0.08111944049596786, 0.24344637989997864, -0.0181470587849617], "left_elbow": [0.11824966222047806, 0.12197338044643402, -0.021892601624131203], "left_wrist": [0.14754801988601685, -0.0040277112275362015, 0.014088761992752552], "right_hip": [0.05466757342219353, -0.027295198291540146, -0.015528455376625061], "right_knee": [0.07225559651851654, -0.18235255777835846, -0.018520904704928398], "right_ankle": [0.089942067861557, -0.3677787184715271, -0.019252480939030647], "left_hip": [-0.042525578290224075, -0.03484155982732773, -0.015481928363442421], "left_knee": [-0.06011202931404114, -0.180166095495224, -0.018695630133152008], "left_ankle": [-0.07281138002872467, -0.36362409591674805, -0.018701398745179176], "right_eye": [-0.011531195603311062, 0.329828143119812, 0.0005379004869610071], "left_eye": [0.0317026749253273, 0.33132410049438477, 0.007234722841531038], "right_ear": [-0.03255778178572655, 0.311111181974411, -0.02960335463285446], "left_ear": [0.05058026313781738, 0.30734145641326904, -0.026135355234146118]},
}

def joint_mapper_smplx_to_openpose18(joints):
    indices = np.array([
        56, # nose
        13, # neck
        18, # right_shoulder
        20, # right_elbow
        22, # right_wrist
        17, # left_shoulder
        19, # left_elbow
        21, # left_wrist
        3,  # right_hip
        6,  # right_knee
        9,  # right_ankle
        2,  # left_hip
        5,  # left_knee
        8,  # left_ankle
        57, # right_eye
        58, # left_eye
        59, # right_ear
        60, # left_ear
    ], dtype=np.int64) - 1
    return joints[indices]

class Skeleton:

    def __init__(self):

        # init pose [18, 3], in [-1, 1]^3
        self.points3D = np.array([
            [-0.00313026,  0.16587697,  0.05414092],
            [-0.00857283,  0.1093518 , -0.00522604],
            [-0.06817748,  0.10397182, -0.00657925],
            [-0.11421658,  0.04033477,  0.00040599],
            [-0.15643744, -0.02915882,  0.03309248],
            [ 0.05288884,  0.10729481, -0.00067854],
            [ 0.10355149,  0.04464601, -0.00735265],
            [ 0.15390812, -0.02282556,  0.03085238],
            [ 0.03897187, -0.0403506 ,  0.00220192],
            [ 0.04027461, -0.15746351, -0.00187036],
            [ 0.04605377, -0.26837209, -0.0018945 ],
            [-0.0507806 , -0.04887162,  0.0022531 ],
            [-0.04873568, -0.16551849, -0.00128197],
            [-0.04840493, -0.27510208, -0.00128831],
            [-0.03098677,  0.19395538,  0.01987491],
            [ 0.01657042,  0.19560097,  0.02724142],
            [-0.05411603,  0.17336673, -0.01328044],
            [ 0.03733583,  0.16922003, -0.00946565]
        ], dtype=np.float32)

        self.name = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]

        # homogeneous
        self.points3D = np.concatenate([self.points3D, np.ones_like(self.points3D[:, :1])], axis=1) # [18, 4]

        # lines [17, 2]
        self.lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]], dtype=np.int32)

        # keypoint color [18, 3]
        # color as in controlnet_aux (https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/open_pose/util.py#L94C5-L96C73)
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        # smplx mesh if available
        self.smplx_model = None
        self.vertices = None
        self.faces = None
        self.ori_center = None
        self.ori_scale = None

        self.body_pose = np.zeros((21, 3), dtype=np.float32)
        # let's default to A-pose
        self.body_pose[15, 2] = -0.7853982
        self.body_pose[16, 2] = 0.7853982
        self.body_pose[0, 1] = 0.2
        self.body_pose[0, 2] = 0.1
        self.body_pose[1, 1] = -0.2
        self.body_pose[1, 2] = -0.1
        """ SMPLX body_pose definition
        0: 'left_hip',#'L_Hip', XYZ -> (-X)(-Y)Z, 后外高 -> 前里高 (3) XYZ
        1: 'right_hip',#'R_Hip', (4) XYZ -> (-X)(-Y)Z, 后里低 -> 前外低 (4) XYZ
        2: 'spine1',#'Spine1', (-X)Y(-Z) -> (0) XYZ
        3: 'left_knee',#'L_Knee', 同左UpperLeg
        4: 'right_knee',#'R_Knee',同右UpperLeg
        5: 'spine2',
        6: 'left_ankle',
        7: 'right_ankle',#'R_Ankle',同右UpperLeg
        8: 'spine3',#'Spine3', (-X)Y(-Z) 同脊椎
        9: 'left_foot',#'L_Foot',同左UpperLeg
        10: 'right_foot',#'R_Foot',同右UpperLeg
        11: 'neck',#'Neck', (-X)Y(-Z) 同脊椎
        12: 'left_collar',#'L_Collar', XYZ -> ZXY (VRM), 前拧, 后, 高 -> 高, 前拧, 后 (1) YZX
        13: 'right_collar',#'R_Collar', XYZ -> (-Z)(-X)Y , 前拧, 前, 低 -> 高, 后拧, 前 (2) YZX
        14: 'head',#'Head', (-X)Y(-Z) 同脊椎
        15: 'left_shoulder',#'L_Shoulder', 同左肩膀
        16: 'right_shoulder',#'R_Shoulder', 同右肩膀
        17: 'left_elbow',#'L_Elbow', 同左肩膀
        18: 'right_elbow',#'R_Elbow', 同右肩膀
        19: 'left_wrist',#'L_Wrist', 同左肩膀
        20: 'right_wrist',#'R_Wrist', 同右肩膀
        """

        self.left_hand_pose = np.zeros((15, 3), dtype=np.float32)
        self.right_hand_pose = np.zeros((15, 3), dtype=np.float32)
        """ hand_pose definition
        index, middle, pinky, ring, thumb; each with 3 joints.
        """
    
    @property
    def center(self):
        return self.points3D[:, :3].mean(0)
    
    @property
    def center_upper(self):
        return self.points3D[0, :3]

    @property
    def torso_bbox(self):
        # valid_points = self.points3D[[0, 1, 8, 11], :3]
        valid_points = self.points3D[:, :3]
        # assure 3D thickness
        min_point = valid_points.min(0) - 0.1
        max_point = valid_points.max(0) + 0.1
        remedy_thickness = np.maximum(0, 0.8 - (max_point - min_point)) / 2
        min_point -= remedy_thickness
        max_point += remedy_thickness
        return min_point, max_point

    def sample_points(self, noise=0.05, N=1000):
        # just sample N points around each line
        pc = []
        for i in range(17):
            A = self.points3D[[self.lines[i][0]], :3] # [1, 3]
            B = self.points3D[[self.lines[i][1]], :3]
            x = np.linspace(0, 1, N)[:, None] # [N, 1]
            points = A * (1 - x) + B * x
            # add noise
            points += np.random.randn(N, 3) * noise
            pc.append(points)
        pc = np.concatenate(pc, axis=0) # [17 * N, 3]
        return pc
    
    def write_json(self, path):

        with open(path, 'w') as f:
            d = {}
            for i in range(18):
                d[self.name[i]] = self.points3D[i, :3].tolist()
            json.dump(d, f)
    
    def load_json(self, path):
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                d = json.load(f)
        else:
            # assume it's a preset
            d = PRESET[path]
        
        # load keypoints
        for i in range(18):
            self.points3D[i, :3] = np.array(d[self.name[i]])
    
    def load_smplx(self, path, betas=None, expression=None, gender='neutral'):

        import smplx

        if self.smplx_model is None:

            self.smplx_model = smplx.create(
                path, 
                model_type='smplx',
                gender=gender, 
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext='npz',
                use_pca=False, # explicitly control hand pose
                flat_hand_mean=True, # use a flatten hand default pose
            )

        # betas = torch.randn([1, self.smplx_model.num_betas], dtype=torch.float32)
        # expression = torch.randn([1, self.smplx_model.num_expression_coeffs], dtype=torch.float32)

        smplx_output = self.smplx_model(
            body_pose=torch.tensor(self.body_pose, dtype=torch.float32).unsqueeze(0),
            left_hand_pose=torch.tensor(self.left_hand_pose, dtype=torch.float32).unsqueeze(0),
            right_hand_pose=torch.tensor(self.right_hand_pose, dtype=torch.float32).unsqueeze(0),
            betas=betas, expression=expression, return_verts=True
        )

        self.vertices = smplx_output.vertices.detach().cpu().numpy()[0] # [10475, 3]
        self.faces = self.smplx_model.faces # [20908, 3]

        # tmp: save deformed smplx mesh
        # import trimesh
        # _mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        # _mesh.export('smplx.obj')

        joints = smplx_output.joints.detach().cpu().numpy()[0] # [127, 3]
        joints = joint_mapper_smplx_to_openpose18(joints)

        self.points3D = np.concatenate([joints, np.ones_like(joints[:, :1])], axis=1) # [18, 4]

        # rescale and recenter 
        if self.ori_center is None:
            vmin = self.vertices.min(0)
            vmax = self.vertices.max(0)
            self.ori_center = (vmax + vmin) / 2
            self.ori_scale = 0.8 / np.max(vmax - vmin)
        
        self.vertices = (self.vertices - self.ori_center) * self.ori_scale
        self.points3D[:, :3] = (self.points3D[:, :3] - self.ori_center) * self.ori_scale

        
    def scale(self, delta):
        self.points3D[:, :3] *= 1.1 ** (-delta)
        if self.vertices is not None:
            self.vertices *= 1.1 ** (-delta)

    def pan(self, rot, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        delta = 0.0005 * rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        self.points3D[:, :3] += delta
        if self.vertices is not None:
            self.vertices += delta

    def draw(self, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = self.points3D @ mvp.T # [18, 4]
        points = points[:, :3] / points[:, 3:] # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H # [18]
        ys = (points[:, 1] + 1) / 2 * W # [18]
        mask = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)

        # hide certain keypoints based on empirical occlusion
        if enable_occlusion:
            # decide view by the position of nose between two ears
            if points[0, 2] > points[-1, 2] and points[0, 2] < points[-2, 2]:
                # left view
                mask[-2] = False # no right ear
                if xs[-4] > xs[-3]:
                    mask[-4] = False # no right eye if it's "righter" than left eye
            elif points[0, 2] < points[-1, 2] and points[0, 2] > points[-2, 2]:
                # right view
                mask[-1] = False
                if xs[-3] < xs[-4]:
                    mask[-3] = False
            elif points[0, 2] > points[-1, 2] and points[0, 2] > points[-2, 2]:
                # back view
                mask[0] = False # no nose
                mask[-3] = False # no eyes
                mask[-4] = False

        # 18 points
        for i in range(18):
            if not mask[i]: continue
            cv2.circle(canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1)

        # 17 lines
        for i in range(17):
            cur_canvas = canvas.copy()
            if not mask[self.lines[i]].all(): 
                continue
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
            
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)
        

class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.skel = Skeleton()
        self.glctx = dr.RasterizeCudaContext()
        
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation

        self.save_image_path = 'pose.png'
        self.save_json_path = 'pose.json'
        self.mouse_loc = np.array([0, 0])
        self.points2D = None # [18, 2]
        self.point_idx = 0
        self.drag_sensitivity = 0.0001
        self.pan_scale_skel = True
        self.enable_occlusion = True
        
        dpg.create_context()
        self.register_dpg()
        self.step()
        

    def __del__(self):
        dpg.destroy_context()


    def step(self):

        if self.need_update:
        
            # mvp
            mv = self.cam.view # [4, 4]
            proj = self.cam.perspective # [4, 4]
            mvp = proj @ mv

            # render our openpose image, somehow
            self.render_buffer, self.points2D = self.skel.draw(mvp, self.H, self.W, enable_occlusion=self.enable_occlusion)

            # if with smplx, overlay normal of mesh
            if self.skel.vertices is not None:
                mesh_normal = self.render_mesh_normal(mvp, self.H, self.W, self.skel.vertices, self.skel.faces)
            
                # mix normal with buffer
                self.render_buffer = self.render_buffer + mesh_normal
        
            self.need_update = False
            
            dpg.set_value("_texture", self.render_buffer)

    def render_mesh_normal(self, mvp, H, W, vertices, faces):

        mvp = torch.from_numpy(mvp.astype(np.float32)).cuda()
        vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
        faces = torch.from_numpy(faces.astype(np.int32)).cuda()

        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0) # [1, N, 4]
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, (H, W))

        i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
        v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(vertices)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = safe_normalize(normal)

        normal_image = (normal[0] + 1) / 2
        normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
        buffer = normal_image.detach().cpu().numpy()

        return buffer

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(label="Viewer", tag="_primary_window", width=self.W, height=self.H, pos=[0, 0], no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=600, height=self.H, pos=[self.W, 0], no_move=True, no_title_bar=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # save image    
            def callback_save_image(sender, app_data):
                image = (self.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_image_path, image)
                print(f'[INFO] write image to {self.save_image_path}')
            
            def callback_set_save_image_path(sender, app_data):
                self.save_image_path = app_data
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)
                dpg.bind_item_theme("_button_save_image", theme_button)

                dpg.add_input_text(label="", default_value=self.save_image_path, callback=callback_set_save_image_path)
            
            # save json
            def callback_save_json(sender, app_data):
                self.skel.write_json(self.save_json_path)
                print(f'[INFO] write json to {self.save_json_path}')
            
            def callback_set_save_json_path(sender, app_data):
                self.save_json_path = app_data
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="save json", tag="_button_save_json", callback=callback_save_json)
                dpg.bind_item_theme("_button_save_json", theme_button)

                dpg.add_input_text(label="", default_value=self.save_json_path, callback=callback_set_save_json_path)

            # pan/scale mode
            def callback_set_pan_scale_mode(sender, app_data):
                self.pan_scale_skel = not self.pan_scale_skel

            dpg.add_checkbox(label="pan/scale skeleton", default_value=self.pan_scale_skel, callback=callback_set_pan_scale_mode)

            # backview mode
            def callback_set_occlusion_mode(sender, app_data):
                self.enable_occlusion = not self.enable_occlusion
                self.need_update = True

            dpg.add_checkbox(label="use occlusion", default_value=self.enable_occlusion, callback=callback_set_occlusion_mode)

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

            # drag sensitivity
            def callback_set_drag_sensitivity(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_float(label="drag sensitivity", min_value=0.000001, max_value=0.001, format="%f", default_value=self.drag_sensitivity, callback=callback_set_drag_sensitivity)
            
            # SMPLX pose editing
            with dpg.collapsing_header(label="SMPLX body_pose", default_open=False):

                def callback_update_body_pose(sender, app_data, user_data):
                    self.skel.body_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.body_pose.shape[0]):
                    dpg.add_input_floatx(default_value=self.skel.body_pose[i], size=3, width=200, format="%.3f", on_enter=False, callback=callback_update_body_pose, user_data=i)
            
            with dpg.collapsing_header(label="SMPLX left_hand_pose", default_open=False):

                def callback_update_left_hand_pose(sender, app_data, user_data):
                    self.skel.left_hand_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.left_hand_pose.shape[0]):
                    dpg.add_input_floatx(default_value=self.skel.left_hand_pose[i], size=3, width=200, format="%.3f", on_enter=False, callback=callback_update_left_hand_pose, user_data=i)
            
            with dpg.collapsing_header(label="SMPLX right_hand_pose", default_open=False):

                def callback_update_right_hand_pose(sender, app_data, user_data):
                    self.skel.right_hand_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.right_hand_pose.shape[0]):
                    dpg.add_input_floatx(default_value=self.skel.right_hand_pose[i], size=3, width=200, format="%.3f", on_enter=False, callback=callback_update_right_hand_pose, user_data=i)


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data
            
            if self.pan_scale_skel:
                self.skel.scale(delta)
            else:
                self.cam.scale(delta)

            self.need_update = True


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            if self.pan_scale_skel:
                self.skel.pan(self.cam.rot, dx, dy)
            else:
                self.cam.pan(dx, dy)

            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        def callback_skel_select(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            
            # determine the selected keypoint from mouse_loc
            if self.points2D is None: return # not prepared

            dist = np.linalg.norm(self.points2D - self.mouse_loc, axis=1) # [18]
            self.point_idx = np.argmin(dist)

        
        def callback_skel_drag(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]
        
            self.skel.points3D[self.point_idx, :3] += self.drag_sensitivity * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])

            self.need_update = True


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

            # for skeleton editing
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_skel_select)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_skel_drag)

        
        dpg.create_viewport(title='pose viewer', resizable=False, width=self.W + 600, height=self.H)
        
        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.focus_item("_primary_window")

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--load', type=str, default=None, help="path to load a json pose, or a preset name (2head, 2.5head, 3head, 4head)")
    parser.add_argument('--smplx_path', type=str, default=None, help="path to models folder (contains smplx/)")
    parser.add_argument('--save', type=str, default=None, help="path to render and save pose images")
    parser.add_argument('--radius', type=float, default=2.7, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=18.8, help="default GUI camera fovy")

    opt = parser.parse_args()

    gui = GUI(opt)

    if opt.smplx_path is not None:
        print(f'[INFO] load smplx from {opt.smplx_path}')
        gui.skel.load_smplx(opt.smplx_path)
        gui.need_update = True

    if opt.load is not None:
        print(f'[INFO] load from {opt.load}')
        gui.skel.load_json(opt.load)
        gui.need_update = True
    
    if opt.save is not None:
        os.makedirs(opt.save, exist_ok=True)
        # render from fixed views and save all images
        elevation = [0, 10, 20]
        azimuth = np.arange(0, 360, dtype=np.int32)
        for ele in tqdm.tqdm(elevation):
            for azi in tqdm.tqdm(azimuth):
                gui.cam.from_angle(ele, azi)
                gui.need_update = True
                gui.step()
                dpg.render_dearpygui_frame()
                image = (gui.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(opt.save, f'{ele}_{azi:04d}.jpg'), image)
    else:
        gui.render()