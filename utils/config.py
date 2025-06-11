
class Config:
    def __init__(self):
        self.final_conv_kernel = 1

        self.dataset = {
            'data_dir': './Dataset/300w/images',
            'train': './Dataset/300w/train.csv',
            'test': './Dataset/300w/valid.csv',
            'rot_factor': 30,
            'scale_factor': 0.25,

        }

        self.model = {
            'sigma': 1,
            'num_joints': 68,
            'image_size': (256, 256),
            'heatmap_size': (64, 64),
            'pretrained': 'weights/imagenet.pth'

        }
        self.stages = {
            'stage2': {
                'num_modules': 1,
                'num_branches': 2,
                'num_blocks': [4, 4],
                'num_channels': [18, 36],
                'block': 'BASIC',
                'fuse_method': 'SUM'
            },
            'stage3': {
                'num_modules': 4,
                'num_branches': 3,
                'num_blocks': [4, 4, 4],
                'num_channels': [18, 36, 72],
                'block': 'BASIC',
                'fuse_method': 'SUM'
            },
            'stage4': {
                'num_modules': 3,
                'num_branches': 4,
                'num_blocks': [4, 4, 4, 4],
                'num_channels': [18, 36, 72, 144],
                'block': 'BASIC',
                'fuse_method': 'SUM'
            }
        }

        self.train = {
            'batch_size': 16,
            'epochs': 120,
            'num_lms': 68,
        }

