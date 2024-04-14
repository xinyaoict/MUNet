import os.path
import logging
import torch
from utils import utils_logger
from utils import utils_image as util
from utils import utils_model

def main():

    testset_name = 'Real'  # folder name of real images
    n_channels = 3            # set 1 for grayscale image, set 3 for color image
    model_name = 'model.pth'
    nc = [64,128,256,512]
    nb = 4
    testsets = 'testsets'    
    results = 'test_results'     

    result_name = testset_name + '_' + model_name[:-4]
    L_path = os.path.join(testsets, testset_name)
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    model_pool = 'model_zoo'  # fixed
    model_path = os.path.join(model_pool, model_name)
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        print(f'No model found at {model_path}')
    
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    border = 0


    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_munet import MUNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='BR')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
     v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))


    L_paths = util.get_image_paths(L_path)
    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
       
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        img_E = utils_model.test_mode(model, img_L, mode=3)
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)
        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

if __name__ == '__main__':
    main()
