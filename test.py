import os
from collections import OrderedDict
from skimage import io
import shutil

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.labeler import Labeler
from util.preprocess_loader import Loader
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

# +-----------------------------------------+
# |             PRE-PROCESSING              |
# +-----------------------------------------+

# Load Images
loader = Loader(opt)
images, color_metas = loader.load()

# Load Labeler Class
labeler = Labeler(opt)

# For each file
for image, color_meta, file in zip(images, color_metas, loader.files):
    if not (os.path.exists(os.path.join(opt.inst_path, file)) and
            os.path.exists(os.path.join(opt.label_path, file)) and
            os.path.exists(os.path.join(opt.style_path, file.replace('png', 'jpg')))):
        labeled_img = labeler.label(image, color_meta)
        io.imsave(os.path.join(opt.inst_path, file), labeled_img)
        io.imsave(os.path.join(opt.label_path, file), labeled_img)

        # @TODO: Temporary fix, Implement Dynamic Nature
        shutil.copyfile(os.path.join(opt.style_set_path, '7.jpg'),
                        os.path.join(opt.style_path, file.replace('png', 'jpg')))

# +-----------------------------------------+
# |               TESTING                   |
# +-----------------------------------------+

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)

model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    '''
    data_i contains 4 parameters
    1. label
    2. instance     (
    3. image        (RGB Image cropped to the size and normalized -1 to 1)
    4. path         (string: specify the image location)
    '''
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    print('saving')
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
