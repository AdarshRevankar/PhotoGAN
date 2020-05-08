from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./static/output/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)

        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False

        # +-----------------------------------------------------+
        # |             PRE-PROCESSING PARAMS                   |
        # +-----------------------------------------------------+

        # Image Params
        parser.add_argument('--width', type=int, default=256, help='Specifies the width of the input image')
        parser.add_argument('--height', type=int, default=256, help='Specifies the width of the output image')
        parser.add_argument('--color', type=str, default='rgb', help='Color Encoding of image [rgb|label]')

        # Path
        parser.add_argument('--drawing_file', type=str, default='None', help='Provides the drawing image name')
        parser.add_argument('--style_index', type=int, default=1,
                            help='Style Index number specifying which style image was clicked')
        parser.add_argument('--drawings_path', type=str, default='datasets/coco_stuff/val_drawing',
                            help='Input Path for Pre-processing')
        parser.add_argument('--style_set_path', type=str, default='datasets/coco_stuff/val_styles',
                            help='Input Path for Determining Styles')

        parser.add_argument('--inst_path', type=str, default='datasets/coco_stuff/val_inst',
                            help='Instance Image storage path')
        parser.add_argument('--label_path', type=str, default='datasets/coco_stuff/val_label',
                            help='Label Image Storage path')
        parser.add_argument('--style_path', type=str, default='datasets/coco_stuff/val_img',
                            help='Style Image Storage Path')

        parser.add_argument('--color_code_path', type=str, default='datasets/coco_stuff/color_code.csv',
                            help='Contains color to label map info')
        return parser
