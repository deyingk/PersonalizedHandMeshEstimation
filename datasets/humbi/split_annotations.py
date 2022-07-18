"""
Split annotations.

small:      only use first 100 subjects
split_1:    80/20 randomly split all images into training and test sets
            training set and test set may contain same subjects
split_2:    split the dataset regarding subjects 80/20
            training set and test set have no common subject

split_3:    for each sequence, first 80% as training, last 20% images as test.
            training iamges 420628
            test images 104738

split_4:    split the dataset regarding subjects
            training set and test set have no common subject   9/1 split

split_5:    split the dataset regarding subjects 90/10
            training set and test set have no common subjects


split_51:
            split the dataset using 'split_5',
            split the training set further to 5-fold 'cross-validation'
            split_55  [train][train][train][train][test]
            split_54  [train][train][train][test][train]
            split_53  [train][train][test][train][train]
            split_52  [train][test][train][train][train]
            split_51  [test][train][train][train][train]

split_6:    split the dataset regarding subjects 90/10
            training set and test set have no common subjects
            the subjects are sorted and the smallest and biggest hands are chosen into test set.
"""

import os
import pickle
import random
from collections import defaultdict

random.seed(0)

small_set = False
which_split = 'split_55'


annotation_dir = '../../data/humbi/annotations' 
split_annotation_dir = '../../data/humbi/split_annotations'

all_annotation_files = os.listdir(annotation_dir)
all_subjects = sorted([int(anno_name.strip('subject_').strip('_anno.pkl')) for anno_name in all_annotation_files])
all_subjects = list(map(str, all_subjects))

if not os.path.isdir(split_annotation_dir):
    os.makedirs(split_annotation_dir)

annotation_list = []

if which_split == 'split_1':
    if small_set:
        subject_list = all_subjects[:len(all_subjects)//4]
        for subject in subject_list:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            annotation_list.extend(subject_anno)
        random.shuffle(annotation_list)
        total_length = len(annotation_list)
        training_anno = annotation_list[: int(0.8*total_length)]
        test_anno = annotation_list[int(0.8*total_length):]

        with open(os.path.join(split_annotation_dir, 'training_small_split_1.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        with open(os.path.join(split_annotation_dir, 'test_small_split_1.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)
            
    else:
        subject_list = all_subjects[:]
        for subject in subject_list:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            annotation_list.extend(subject_anno)
        random.shuffle(annotation_list)
        total_length = len(annotation_list)
        training_anno = annotation_list[: int(0.8*total_length)]
        test_anno = annotation_list[int(0.8*total_length):]

        with open(os.path.join(split_annotation_dir, 'training_all_split_1.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        with open(os.path.join(split_annotation_dir, 'test_all_split_1.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)

elif which_split == 'split_2':
    if small_set:
        subject_list = all_subjects[:len(all_subjects)//4]
        print(len(subject_list))
        ratio = 0.85

        training_anno = []
        for subject in subject_list[:int(len(subject_list)*ratio)]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            training_anno.extend(subject_anno)
        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_small_split_2.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        test_anno = []
        for subject in subject_list[int(len(subject_list)*ratio):]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            test_anno.extend(subject_anno)
        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_small_split_2.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)
    else:
        subject_list = all_subjects
        print(len(subject_list))
        random.shuffle(subject_list)
        ratio = 0.8

        training_anno = []
        for subject in subject_list[:int(len(subject_list)*ratio)]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            training_anno.extend(subject_anno)
        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_all_split_2.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        test_anno = []
        for subject in subject_list[int(len(subject_list)*ratio):]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            test_anno.extend(subject_anno)
        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_all_split_2.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)

elif which_split == 'split_3':
    if small_set:
        raise "Not implemented yet!"

    else:
        subject_list = all_subjects
        print(len(subject_list))
        ratio = 0.8

        training_anno = []
        test_anno = []
        for subject in subject_list:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            dictionary = defaultdict(list)
            for anno_i in subject_anno:
                img_path  = anno_i[0]
                camera = img_path.split('/')[2]
                dictionary[camera].append(anno_i)
            for key, value in dictionary.items():
                training_anno.extend(value[:round(len(value) * ratio)])
                test_anno.extend(value[round(len(value) * ratio):])

        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_all_split_3.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_all_split_3.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)

elif which_split == 'split_4':
    if small_set:
        raise "Not implemented yet!"

    else:
        subject_list = all_subjects
        print(subject_list)
        # stop
        print(len(subject_list))
        # random.shuffle(subject_list)
        ratio = 0.9

        training_anno = []
        for subject in subject_list[:int(len(subject_list)*ratio)]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            training_anno.extend(subject_anno)
        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_all_split_4.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        test_anno = []
        for subject in subject_list[int(len(subject_list)*ratio):]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            test_anno.extend(subject_anno)
        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_all_split_4.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)

elif which_split == 'split_5':
    if small_set:
        raise "Not implemented!"
    else:
        subject_list = all_subjects
        print(len(subject_list))
        random.shuffle(subject_list)
        ratio = 0.9

        stop

        training_anno = []
        for subject in subject_list[:int(len(subject_list)*ratio)]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            training_anno.extend(subject_anno)
        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_all_split_5.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        test_anno = []
        for subject in subject_list[int(len(subject_list)*ratio):]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            test_anno.extend(subject_anno)
        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_all_split_5.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)

elif which_split in ('split_51', 'split_52', 'split_53','split_54','split_55'):
    if small_set:
        raise "Not implemented!"
    else:
        subject_list = all_subjects
        random.shuffle(subject_list)
        ratio = 0.9
        print(len(subject_list[int(len(subject_list)*ratio):]))
        print('excluded subjects are, ', sorted(subject_list[int(len(subject_list)*ratio):],key=int))
        print('\n')
        subject_list = subject_list[:int(len(subject_list)*ratio)]
        number_subjects = int(len(subject_list)*ratio)
        print(subject_list)
        
        which_cross = int(which_split[-1])
        print('which_cross', which_cross)

        training_anno = []
        test_anno = []
        test_subjects = []

        for i, subject in enumerate(subject_list):
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)

            if i >= (which_cross-1)/5*number_subjects and i< which_cross/5*number_subjects:
                test_anno.extend(subject_anno)                      
                test_subjects.append(subject)
            else:
                training_anno.extend(subject_anno)
        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_all_'+ which_split+'.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)
        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_all_'+ which_split+'.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)
        print('test subjects includes,', test_subjects)

elif which_split == 'split_6':
    if small_set:
        raise "Not implemented!"
    else:
        subject_list = all_subjects
        print(len(subject_list))
        random.shuffle(subject_list)
        ratio = 0.9

        stop

        training_anno = []
        for subject in subject_list[:int(len(subject_list)*ratio)]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            training_anno.extend(subject_anno)
        random.shuffle(training_anno)
        with open(os.path.join(split_annotation_dir, 'training_all_split_5.pkl'), 'wb') as f:
            pickle.dump(training_anno, f)

        test_anno = []
        for subject in subject_list[int(len(subject_list)*ratio):]:
            with open(os.path.join(annotation_dir, 'subject_'+subject+'_anno.pkl'), 'rb') as f:
                subject_anno = pickle.load(f)
            test_anno.extend(subject_anno)
        random.shuffle(test_anno)
        with open(os.path.join(split_annotation_dir, 'test_all_split_5.pkl'), 'wb') as f:
            pickle.dump(test_anno, f)

else:
    raise 'Not implemented yet!'

# print(len(annotation_list))
print(len(training_anno))
print(len(test_anno))

