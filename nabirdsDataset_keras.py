import os
import sys
import keras
import pydot
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt


dataset_path = 'nabirds-data/nabirds/'
image_path = dataset_path + 'images/'



def load_images(dir=image_path, labels="inferred", batch_size=32, img_size=(256, 256), seed=0, split=0.8):
    training, validation = keras.utils.image_dataset_from_directory(
                directory = dir,
                labels = labels,
                label_mode = "categorical",
                class_names = None,
                color_mode = "rgb",
                batch_size = batch_size,
                image_size = img_size,
                shuffle = True, 
                seed = seed,
                validation_split = split,
                subset = "both",
                interpolation = "bilinear",
                follow_links = False,
                crop_to_aspect_ratio = False,
                pad_to_aspect_ratio = False,
                data_format = None,
                verbose = True,
            )

    return training, validation



def create_hierarchical_labels(file='nabirds-data/nabirds/hierarchy.txt', draw=False):
    G = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=int)
    G = G.reverse()

    if draw:
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos)
        plt.show()

    topological_order = list(nx.topological_sort(G))
    root = topological_order[0]

    hierarchical_classes = list(G.successors(root))
    leaf_classes = [n for n in G.nodes if G.out_degree(n)==0]

    hierarchy_descendents = {n: nx.descendants(G, n) for n in hierarchical_classes}

    for n in hierarchical_classes:
        hierarchy_descendents[n] = set(leaf_classes) & set(hierarchy_descendents[n])

    correct_dir_labels = np.array(leaf_classes)

    for n in hierarchical_classes:
        for m in hierarchy_descendents[n]:
            idx = np.where(correct_dir_labels == m)
            correct_dir_labels[idx] = n

    correct_labels = []

    for i in range(len(leaf_classes)):
        leaf = leaf_classes[i]
        folder = str(leaf).zfill(4)
        count = len(os.listdir(image_path+folder+'/'))

        correct_labels.extend([correct_dir_labels[i]]*count)

    print(os.walk(image_path))

    np.savetxt('labels/general_class_labels.txt', correct_labels)



def load_images_with_hierarchy(label_dir='labels/general_class_labels.txt', dir=image_path, labels="inferred", batch_size=32, img_size=(256,256), seed=0, split=0.8):
    correct_labels = list(np.loadtxt(label_dir).astype(int))

    training, validate = load_images(dir, correct_labels, batch_size, img_size, seed, split)

    return training, validate

    


if __name__ == "__main__":
    create_hierarchical_labels()
    load_images_with_hierarchy()
