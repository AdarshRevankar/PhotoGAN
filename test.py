from collections import OrderedDict
from skimage import io
import shutil
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.labeler import Labeler
from util.preprocess_loader import Loader
from util.visualizer import Visualizer
import os


def pre_process_semantic_input():
    # Load Images
    images = loader.load()

    # Load Labeler Class
    labeler = Labeler(opt)

    # For each file
    for image, file in zip(images, loader.files):
        if not (os.path.exists(os.path.join(opt.inst_path, file)) and
                os.path.exists(os.path.join(opt.label_path, file)) and
                os.path.exists(os.path.join(opt.style_path, file.replace('png', 'jpg')))):
            # Label The image
            labeled_img = labeler.label(image)

            # Save the Instance Image
            io.imsave(os.path.join(opt.inst_path, file), labeled_img)

            # Save the Label Image
            io.imsave(os.path.join(opt.label_path, file), labeled_img)

            # Copy the Style Image
            shutil.copyfile(
                os.path.join(opt.style_set_path, str(opt.style_index) + '.jpg'),
                os.path.join(opt.style_path, file.replace('png', 'jpg'))
            )


def generate_from_data():
    # test
    for i, data_i in enumerate(data.create_dataloader(opt)):
        if i * opt.batchSize >= opt.how_many:
            break
        generated = model(data_i, mode='inference')
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(img_dir, visuals, img_path[b:b + 1])


def clear_images(max_img_buffer=5, flush=False):
    # Clear the images
    current_images_size = len(loader.files)
    if current_images_size >= max_img_buffer or flush:
        for file in loader.files:
            # Remove Image, Instance, Label
            os.remove(os.path.join(opt.inst_path, file))
            os.remove(os.path.join(opt.label_path, file))
            os.remove(os.path.join(opt.style_path, file.replace('png', 'jpg')))


opt = TestOptions().parse()
model = Pix2PixModel(opt)
model.eval()
visualizer = Visualizer(opt)
img_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
loader = Loader(opt)
