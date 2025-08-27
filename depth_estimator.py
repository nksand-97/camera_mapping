import os
import cv2
import torch


class DepthEstimator:
    canonical_input_size = (616, 1064)
    canonical_focal = 1000
    padding = [123.675, 116.28, 103.53]
    image_norm_mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    image_norm_std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]


    def __init__(self, config):
        # カメラ内部パラメータ
        fu = config["kitti"]["focal_u_cam2"]
        fv = config["kitti"]["focal_v_cam2"]
        cu = config["kitti"]["center_u_cam2"]
        cv = config["kitti"]["center_v_cam2"]
        self.in_param = [fu, fv, cu, cv]

        # 深度推定モデル読み込み
        model_path = config["metric3d"]["model_path"]
        model_name = config["metric3d"]["model_name"]
        ckpt_path = config["metric3d"]["ckpt_path"]
        self.model = torch.hub.load(model_path, model_name, source="local", pretrained=True)
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "weight", ckpt_path+".pth"))["model_state_dict"],
            strict=False
        )
        self.model.cuda().eval()
    

    def estimate(self, rgb_origin):
        self.model.eval()

        rgb_origin = rgb_origin[:, :, ::-1]

        # 入力サイズを学習済みモデルに合わせて調整する
        h, w = rgb_origin.shape[:2]
        scale = min(self.canonical_input_size[0]/h, self.canonical_input_size[1]/w)
        rgb = cv2.resize(rgb_origin, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
        intrinsic = [
            self.in_param[0]*scale, self.in_param[1]*scale, self.in_param[2]*scale, self.in_param[3]*scale
        ]

        # 入力画像を調整したサイズにパディングする
        h, w = rgb.shape[:2]
        pad_h = self.canonical_input_size[0] - h
        pad_w = self.canonical_input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb, pad_h_half, pad_h-pad_h_half, pad_w_half, pad_w_half, cv2.BORDER_CONSTANT, value=self.padding
        )
        pad_info = [pad_h_half, pad_h-pad_h_half, pad_w_half, pad_w-pad_w_half]

        # 入力画像を正規化する
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - self.image_norm_mean), self.image_norm_std)

        # 深度推定を実行する
        rgb = rgb[None, :, :, :].cuda()
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model({"input": rgb})
        
        # パディングを元に戻す
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0]-pad_info[1], pad_info[2] : pred_depth.shape[1]-pad_info[3]
        ]

        # 深度画像を元の入力画像のサイズにアップサンプリングする
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()

        # 標準化された深度情報を実スケールに変換する
        canonical_to_real_scale = intrinsic[0] / self.canonical_focal
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)
        pred_depth = pred_depth.cpu().detach().numpy()

        return pred_depth