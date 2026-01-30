import numpy as np
import os
import glob
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils

# 选择设备为 CPU
device = torch.device('cpu')

gpu_id = 0 # 保留，但实际使用 CPU
dataset = 'phoenix2014' # 支持 [phoenix2014, phoenix2014-T, CSL-Daily]
prefix = './dataset/phoenix2014/phoenix-2014-multisigner' # ['./dataset/CSL-Daily', './dataset/phoenix2014-T', './dataset/phoenix2014/phoenix-2014-multisigner']
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'
model_weights = '/remote-home/cs_cs_zl/HSTENet/SpatioTemporalFusion2/SpatioTemporalFusion2dev_17.30_epoch76_model.pt'  # TODO: 替换为你的路径
select_id = 0 # 选择要显示的视频。539 对应 31October_2009_Saturday_tagesschau_default-8，0 对应 01April_2010_Thursday_heute_default-1，1 对应 01August_2011_Monday_heute_default-6，2 对应 01December_2011_Thursday_heute_default-3

# 加载数据并应用变换
gloss_dict = np.load(dict_path, allow_pickle=True).item()
inputs_list = np.load(f"./preprocess/{dataset}/dev_info.npy", allow_pickle=True).item()
name = inputs_list[select_id]['fileid']
print(f'Generating CAM for {name}')
img_folder = os.path.join(prefix, "features/fullFrame-256x256px/" + inputs_list[select_id]['folder']) if 'phoenix' in dataset else os.path.join(prefix, inputs_list[select_id]['folder'])
img_list = sorted(glob.glob(img_folder))
img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
label_list = []
for phase in inputs_list[select_id]['label'].split(" "):
    if phase == '':
        continue
    if phase in gloss_dict.keys():
        label_list.append(gloss_dict[phase][0])
transform = video_augmentation.Compose([
    video_augmentation.CenterCrop(224),
    video_augmentation.Resize(1.0),
    video_augmentation.ToTensor(),
])
vid, label = transform(img_list, label_list, None)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0)

left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = ['K5', "P2", 'K5', "P2"]
for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride

max_len = vid.size(1)
video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
max_len = max_len + left_pad + right_pad
vid = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    )
    , dim=0).unsqueeze(0)

# 创建保存梯度和特征图的容器
fmap_block = []
grad_block = []

# 定义hook函数
def forward_hook(module, input, output):
    fmap_block.append(output.detach().cpu())

def backward_hook(module, grad_input, grad_output):
    grad_block.append(grad_output[0].detach().cpu())

# 加载模型并注册hook
model = SLRModel(
    num_classes=len(gloss_dict) + 1,
    c2d_type='resnet34',
    conv_type=2,
    use_bn=1,
    gloss_dict=gloss_dict,
    loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0}
)
model.load_state_dict(torch.load(model_weights, map_location=device))
model.eval()
model = model.to(device)  # 确保模型在 CPU 上

if 'phoenix' in dataset:
    target_layer = model.conv2d.corr2.conv_back
else:
    target_layer = model.conv2d.corr3.conv_back

handle_forward = target_layer.register_forward_hook(forward_hook)
handle_backward = target_layer.register_backward_hook(backward_hook)

# 前向传播
with torch.enable_grad():
    vid = vid.to(device)  # 将视频数据移动到 CPU
    video_length = video_length.to(device)
    label = torch.LongTensor([label]).to(device)
    label_lgt = torch.LongTensor([len(label_list)]).to(device)

    ret_dict = model(vid, video_length, label=label, label_lgt=label_lgt)

    # 获取目标类别的概率
    target = ret_dict["sequence_logits"].sum(dim=0)  # 使用整个序列的 logits
    target.backward(torch.ones_like(target))  # 反向传播计算梯度

# 获取特征图和梯度
feature_maps = fmap_block[0].numpy()
grads = grad_block[0].numpy()

# 计算权重
weights = np.mean(grads, axis=(3, 4), keepdims=True)  # 平均空间维度 (N, C, T, 1, 1)

# 生成CAM
cam = np.sum(weights * feature_maps, axis=1)  # (N, T, H, W)
cam = np.maximum(cam, 0)  # ReLU


# 可视化函数（修改后）
def cam_show_img(original_img, cam, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 预处理原始视频帧
    original_img = (original_img[0].permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5  # 反归一化

    for t in range(cam.shape[1]):
        # 获取当前帧的CAM
        current_cam = cam[0, t]
        current_cam = cv2.resize(current_cam, (224, 224))
        current_cam = current_cam - current_cam.min()
        current_cam = current_cam / (current_cam.max() + 1e-8)

        # 获取原始图像
        current_img = original_img[t].astype(np.uint8)

        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * current_cam), cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + current_img * 0.6

        # 保存结果
        cv2.imwrite(os.path.join(out_dir, f'frame_{t:04d}.jpg'), superimposed_img)

# 生成并保存CAM图像
cam_show_img(vid.cpu().numpy(), cam, './CAM_results')

# 清理hook
handle_forward.remove()
handle_backward.remove()