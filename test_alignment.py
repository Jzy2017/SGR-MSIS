import os
from H_model import H_estimator,VGG1,VGG2
import torch
import cv2
from dataset import Image_stitch_test
import time
from recon import GCNRecon
import argparse
parser = argparse.ArgumentParser("GCN-Stitching")
parser.add_argument('--data_root', type=str, default='example/', help='location of the dataset')
parser.add_argument('--save_name', type=str, default='example', help='name for saving outputs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--device', type=int, default=0, help='device index if using cuda, else -1')
args = parser.parse_args()
if args.device!=-1:
    use_cuda=True
else:
    use_cuda=False
out_folder = os.path.join('output/', args.save_name)
if not os.path.exists(os.path.join(out_folder)):
    os.makedirs(os.path.join(out_folder))
dataset=Image_stitch_test(ir1_path=os.path.join(args.data_root,'ir_input1'),\
                  ir2_path=os.path.join(args.data_root,'ir_input2'),\
                  vis1_path=os.path.join(args.data_root,'vis_input1'),\
                  vis2_path=os.path.join(args.data_root,'vis_input2'))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)
netR = H_estimator(batch_size=args.batch_size,device=args.device,is_training=False)
net_f = VGG1()
net_ir = VGG2()
net_vis = VGG2()
netS = GCNRecon()


net_ir_path = 'snapshot/latest_I.pkl'
net_vis_path = 'snapshot/latest_V.pkl'
netR_path = 'snapshot/latest_R.pkl'
net_f_path = 'snapshot/latest_F.pkl'
netS_path = 'snapshot/latest_S.pkl' 

if net_f_path is not None:
    net_f.load_state_dict(torch.load(net_f_path,map_location='cpu'))
if net_ir_path is not None:
    net_ir.load_state_dict(torch.load(net_ir_path,map_location='cpu'))
if net_vis_path is not None:
    net_vis.load_state_dict(torch.load(net_vis_path,map_location='cpu'))
if netR_path is not None:
    netR.load_state_dict(torch.load(netR_path,map_location='cpu'))
if netS_path is not None:
    netS.load_state_dict(torch.load(netS_path, map_location='cpu'))
if use_cuda:
    netR = netR.to(args.device)
    net_ir = net_ir.to(args.device)
    net_vis = net_vis.to(args.device)
    net_f = net_f.to(args.device)
    netS = netS.to(args.device)

netR.eval()
net_vis.eval()
net_ir.eval()
netS.eval()

for i,(ir_input1, ir_input2, vis_input1, vis_input2, size, name) in enumerate(dataloader):
    print(name[0])
    if use_cuda:
        ir_input1 = ir_input1.to(args.device)
        ir_input2 = ir_input2.to(args.device)
        vis_input1 = vis_input1.to(args.device)
        vis_input2 = vis_input2.to(args.device)
        size = size.to(args.device)
    train_ir_inputs = torch.cat((ir_input1, ir_input2), 3)
    train_vis_inputs = torch.cat((vis_input1, vis_input2), 3)
    start=time.time()
    with torch.no_grad():
        feature_ir1 = net_f(torch.nn.functional.interpolate(train_ir_inputs[...,0:3].permute(0,3,1,2), [128,128]).float())
        feature_ir2 = net_f(torch.nn.functional.interpolate(train_ir_inputs[...,3:6].permute(0,3,1,2), [128,128]).float())
        feature_vis1 = net_f(torch.nn.functional.interpolate(train_vis_inputs[...,0:3].permute(0,3,1,2), [128,128]).float())
        feature_vis2 = net_f(torch.nn.functional.interpolate(train_vis_inputs[...,3:6].permute(0,3,1,2), [128,128]).float())
        feature_ir1_ = net_f(train_ir_inputs[...,0:3].permute(0,3,1,2).float())
        feature_ir2_ = net_f(train_ir_inputs[...,3:6].permute(0,3,1,2).float())
        feature_vis1_ = net_f(train_vis_inputs[...,0:3].permute(0,3,1,2).float())
        feature_vis2_ = net_f(train_vis_inputs[...,3:6].permute(0,3,1,2).float())

        feature_ir1s = net_ir(feature_ir1)
        feature_ir2s = net_ir(feature_ir2)
        feature_vis1s = net_vis(feature_vis1)
        feature_vis2s = net_vis(feature_vis2)
       
        shift,  ir_warp, vis_warp, ir_f_warp, vis_f_warp = netR(feature_ir1s,feature_ir2s,train_ir_inputs, feature_vis1s,feature_vis2s,train_vis_inputs,size,\
            feature_ir1_,feature_ir2_,feature_vis1_,feature_vis2_)

        ir_f_warp_=torch.mean(ir_f_warp,1)
        recon = netS(ir_f_warp, vis_f_warp)
    recon=recon[0].permute(1,2,0).detach().cpu().numpy()*255
    ir_f_warp_=ir_f_warp_.permute(1,2,0).detach().cpu().numpy()*255

    
    cv2.imwrite(os.path.join(out_folder,name[0]),cv2.cvtColor(recon,cv2.COLOR_RGB2BGR))

