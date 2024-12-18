import nabirds
from nabirdsDataset import nabirdsDataset
import time


def load_data(dataset_path, image_path):
    ### Data ###
    # Image Data
    # image_paths        - dictionary - {image_id: image_path}
    # image_sizes        - dictionary - {image_id: [width, height]}
    # image_bboxes       - dictionary - {image_id: bbox}
    # image_parts        - dictionary - {image_id: [part_id] = [x, y]}
    # image_class_labels - dictionary - {imqage_id: class_id}
    #
    # Class Data
    # class_names        - dictionary - {class_id: class_name}
    # class_hierarchy    - dictionary - {child_id: parent_id}


#
    # Parts Data
    # part_names         - dictionary - {part_id: part_name}
    # part_ids           - list - sorted list of index integers

    # Load image data
    image_paths = nabirds.load_image_paths(dataset_path, image_path)
    image_sizes = nabirds.load_image_sizes(dataset_path)
    image_bboxes = nabirds.load_bounding_box_annotations(dataset_path)
    image_parts = nabirds.load_part_annotations(dataset_path)
    image_class_labels = nabirds.load_image_labels(dataset_path)

    # Load in the class data
    class_names = nabirds.load_class_names(dataset_path)
    class_hierarchy = nabirds.load_hierarchy(dataset_path)
    #print(list(class_hierarchy.values()))

    # Load in the part data
    part_names = nabirds.load_part_names(dataset_path)
    part_ids = list(part_names.keys())
    part_ids.sort() 

    return image_paths, image_sizes, image_bboxes, image_parts, image_class_labels, class_names, class_hierarchy, part_names, part_ids


def load_train_test(dataset_path):
    ### Data ###
    # train_images - list - image_ids for training set
    # test_images  - list - image_ids for testing set

    # Load in the train / test split
    train_images, test_images = nabirds.load_train_test_split(dataset_path)

    print(train_images)
    print(test_images)

    return train_images, test_images


def timer(func):
    def timer_wrap(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()

        print(func.__name__, round(t1 - t0, 3))
        return result

    return timer_wrap()
