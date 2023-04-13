import os
import argparse
import sys
import torch
import torch.onnx
import models

# to execute, run
# python onnxExport.py --loadckpt ./checkpoints/dispnet_model_best.pth.tar

parser = argparse.ArgumentParser(description='DispNetC OnnxExport', fromfile_prefix_chars='@')

parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
args = parser.parse_args()


def main():
    assert os.path.isfile(args.loadckpt), "=> no model found at '{}'".format(args.loadckpt)

    # input
    dummy_l = torch.randn(1, 3, 512, 512) 
    dummy_l = dummy_l.cuda()   
    dummy_r = torch.randn(1, 3, 512, 512)    
    dummy_r = dummy_r.cuda()      
    dummy_input = (dummy_l, dummy_r)
    
    dispNet = models.DispNetS()    
    dispNet = torch.nn.DataParallel(dispNet)
 
    checkpoint = torch.load(args.loadckpt)
   
    # DispNetS from SfmLearner has one input, so check if conv1.0.weight has 3 channel.
    # If so, increae its channels to 6
    #conv10w = checkpoint['state_dict']['conv1.0.weight']
    #if conv10w.size()[1] == 3:
    #    checkpoint['state_dict']['conv1.0.weight'] = torch.cat([conv10w, conv10w], 1)    

    dispNet.load_state_dict(checkpoint['state_dict'])
    
    
    dispNet.eval()
    #print(dispNet.module)
  
    torch.onnx.export(dispNet.module, dummy_input, "dispNet_512x512.onnx",  opset_version=11)


if __name__ == '__main__':
    main()