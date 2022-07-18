import pickle

_SUBJECTS = [
        '20200709-subject-01',
        '20200813-subject-02',
        '20200820-subject-03',
        '20200903-subject-04',
        '20200908-subject-05',
        '20200918-subject-06',
        '20200928-subject-07',
        '20201002-subject-08',
        '20201015-subject-09',
        '20201022-subject-10',
]

split_name = 's11'
if split_name == 's5':
    test_idx = [0, 1]
    val_idx = [3]
    train_idx = [2,4,5,6,7,8,9]

elif split_name == 's6':
    test_idx = [4, 5]
    val_idx = [9]
    train_idx = [0,1,2,3,6,7,8]

elif split_name == 's7':
    test_idx = [2, 9]
    val_idx = [5]
    train_idx = [0,1,3,4,6,7,8]

elif split_name == 's9':
    test_idx = [4,9]
    val_idx = [5]
    train_idx = [0,1,2,3,6,7,8]

elif split_name == 's10':
    test_idx = [2,3]
    val_idx = [0]
    train_idx = [1,4,5,6,7,8,9]

elif split_name == 's11':
    test_idx = [0,2]
    val_idx = [3]
    train_idx = [1,4,5,6,7,8,9]


# get all annotations
annotations = []
with open('../../data/dex_ycb/split_annotations/s1_train_size_2.pkl', 'rb') as f:
    anno = pickle.load(f)
    annotations.extend(anno)

with open('../../data/dex_ycb/split_annotations/s1_val_size_2.pkl', 'rb') as f:
    anno = pickle.load(f)
    annotations.extend(anno)

with open('../../data/dex_ycb/split_annotations/s1_test_size_2.pkl', 'rb') as f:
    anno = pickle.load(f)
    annotations.extend(anno)

print(len(annotations))

train_anno = []
val_anno = []
test_anno = []
for anno in annotations:
    cropped_img_path = anno['cropped_img_path']
    if _SUBJECTS[test_idx[0]] in cropped_img_path or _SUBJECTS[test_idx[1]] in cropped_img_path:
        test_anno.append(anno)
    elif _SUBJECTS[val_idx[0]] in cropped_img_path:
        val_anno.append(anno)
    else:
        train_anno.append(anno)

print('length of train_anno', len(train_anno))
print('length of val_anno', len(val_anno))
print('length of test_anno', len(test_anno))

with open('../../data/dex_ycb/split_annotations/{}_train_size_2.pkl'.format(split_name), 'wb') as f:
    pickle.dump(train_anno, f)
with open('../../data/dex_ycb/split_annotations/{}_val_size_2.pkl'.format(split_name), 'wb') as f:
    pickle.dump(val_anno, f)
with open('../../data/dex_ycb/split_annotations/{}_test_size_2.pkl'.format(split_name), 'wb') as f:
    pickle.dump(test_anno, f)