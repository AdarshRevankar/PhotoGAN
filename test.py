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


def pre_process_semantic_input(style_index=1):
    # Load Images
    loader = Loader(opt)

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
                os.path.join(opt.style_set_path, str(style_index) + '.jpg'),
                os.path.join(opt.style_path, file.replace('png', 'jpg'))
            )


def generate_from_data():
    # test
    dataloader = data.create_dataloader(opt)
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break
        g_loss, generated = model(data_i, mode='inference')
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(img_dir, visuals, img_path[b:b + 1])


def clear_images(exclude_file, max_img_buffer=5, flush=False):
    # Clear the images
    current_images_size = len(os.listdir(opt.drawings_path))
    if current_images_size >= max_img_buffer or flush:
        clear_folder(os.path.join(opt.results_dir + 'coco_pretrained/test_latest/synthesized_image'), exclude_file)
        clear_folder(os.path.join(opt.results_dir + 'coco_pretrained/test_latest/input_label'), exclude_file)
        clear_folder(opt.style_path, exclude_file)
        clear_folder(opt.label_path, exclude_file)
        clear_folder(opt.inst_path, exclude_file)
        clear_folder(opt.drawings_path, exclude_file)


def clear_folder(folder, exclude_file):
    for filename in os.listdir(folder):
        if filename == exclude_file:
            continue
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# Initialize Contents
opt = TestOptions().parse()
model = Pix2PixModel(opt)
model.eval()
visualizer = Visualizer(opt)
img_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
