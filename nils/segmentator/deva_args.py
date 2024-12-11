# Common evaluation arguments
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from deva.model.network import DEVA



home = os.path.join(os.environ["NILS_DIR"], "dependencies")

def add_common_eval_args(parser: ArgumentParser):
    parser.add_argument('--model', default=os.path.join(home,'Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth'))

    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--save_all',
        action='store_true',
        help='Save all frames',
    )
    
    # parser.add_argument("task", default = 0)
    # 
    # 
    parser.add_argument('--multirun', default="task=1")

    # Model parameters
    parser.add_argument('--key_dim', type=int, default=64)
    parser.add_argument('--value_dim', type=int, default=512)
    parser.add_argument('--pix_feat_dim', type=int, default=512)

    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames',
                        help='T_max in XMem, decrease to save memory',
                        type=int,
                        default=16)
    parser.add_argument('--min_mid_term_frames',
                        help='T_min in XMem, decrease to save memory',
                        type=int,
                        default=8)
    parser.add_argument('--max_long_term_elements',
                        help='LT_max in XMem, increase if objects disappear for a long time',
                        type=int,
                        default=20000)
    parser.add_argument('--num_prototypes', help='P in XMem', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every',
                        help='r in XMem. Increase to improve running speed.',
                        type=int,
                        default=8)
    parser.add_argument(
        '--chunk_size',
        default=-1,
        type=int,
        help='''Number of objects to process in parallel as a batch; -1 for unlimited. 
        Set to a small number to save memory.''')

    parser.add_argument(
        '--size',
        default=320,
        type=int,
        help='Resize the shorter side to this size. -1 to use original resolution. ')


# Evaluation arguments for extensions
from argparse import ArgumentParser


def add_ext_eval_args(parser: ArgumentParser):

    # Grounded Segment Anything
    parser.add_argument('--GROUNDING_DINO_CONFIG_PATH',
                        default='./saves/GroundingDINO_SwinT_OGC.py')

    parser.add_argument('--GROUNDING_DINO_CHECKPOINT_PATH',
                        default='./saves/groundingdino_swint_ogc.pth')

    parser.add_argument('--DINO_THRESHOLD', default=0.35, type=float)
    parser.add_argument('--DINO_NMS_THRESHOLD', default=0.8, type=float)

    # Segment Anything (SAM) models
    parser.add_argument('--SAM_ENCODER_VERSION', default='vit_h')
    parser.add_argument('--SAM_CHECKPOINT_PATH', default=os.path.join(home,'Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth'))

    # Mobile SAM
    parser.add_argument('--MOBILE_SAM_CHECKPOINT_PATH', default='./saves/mobile_sam.pt')

    # Segment Anything (SAM) parameters
    parser.add_argument('--SAM_NUM_POINTS_PER_SIDE',
                        type=int,
                        help='Number of points per side for prompting SAM',
                        default=40)
    parser.add_argument('--resegment_whole_image', type=bool, default=True)
    parser.add_argument('--SAM_NUM_POINTS_PER_BATCH',
                        type=int,
                        help='Number of points computed per batch',
                        default=64)
    parser.add_argument('--SAM_PRED_IOU_THRESHOLD',
                        type=float,
                        help='(Predicted) IoU threshold for SAM',
                        default=0.85)
    parser.add_argument('--SAM_OVERLAP_THRESHOLD',
                        type=float,
                        help='Overlap threshold for overlapped mask suppression in SAM',
                        default=0.8)



def add_auto_default_args(parser):
    parser.add_argument('--img_path', default='./example/vipseg')
    parser.add_argument('--detection_every', type=int, default=1)
    parser.add_argument('--num_voting_frames',
                        default=1,
                        type=int,
                        help='Number of frames selected for voting. only valid in semionline')

    parser.add_argument('--temporal_setting', default='semionline', help='semionline/online')
    parser.add_argument('--max_missed_detection_count', type=int, default=5)
    parser.add_argument('--max_num_objects',
                        default=150,
                        type=int,
                        help='Max. num of objects to keep in memory. -1 for no limit')

    parser.add_argument('--sam_variant', default='original', help='mobile/original')
    parser.add_argument('--suppress_small_objects', action='store_true')

    return parser

def add_text_default_args(parser,cfg):
    parser.add_argument('--img_path', default='./example/vipseg')
    parser.add_argument('--detection_every', type=int, default=cfg.deva_detection_every)
    parser.add_argument('--num_voting_frames',
                        default=cfg.deva_n_voting_frames,
                        type=int,
                        help='Number of frames selected for voting. only valid in semionline')
    
    parser.add_argument('--rp_check_interval', type = int, default = 16)

    parser.add_argument('--temporal_setting', default='semionline', help='semionline/online')
    parser.add_argument('--max_missed_detection_count', type=int, default=10)
    parser.add_argument('--max_num_objects',
                        default=-1,
                        type=int,
                        help='Max. num of objects to keep in memory. -1 for no limit')
    parser.add_argument('--prompt', type=str, help='Separate classes with a single fullstop')
    parser.add_argument('--sam_variant', default='original', help='mobile/original')
    return parser


def get_model_and_config(parser: ArgumentParser):
    args = parser.parse_args()
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']

    # Load our checkpoint
    network = DEVA(config).cuda().eval()
    if args.model is not None:
        model_weights = torch.load(args.model)
        network.load_weights(model_weights)
    else:
        print('No model loaded.')

    return network, config, args