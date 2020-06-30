from ..opts import parse_opts
import os



def get_cls(opt, split=1):
    """
    Args:
        opt   : config options
        train : 0 for testing, 1 for training, 2 for validation 
        split : 1,2,3 
    Returns:
        (tensor(frames), class_id ): Shape of tensor C x T x H x W
    """
    train = 0

    lab_names = sorted \
        (set(['_'.join(os.path.splitext(file)[0].split('_')[:-2] )for file in os.listdir(opt.annotation_path)]))

    # Number of classes
    N = len(lab_names)
    assert N == 51

    lab_names = dict(zip(lab_names, range(N)))   # Each label is mappped to a number

    # indexes for training/test set
    split_lab_filenames = sorted([file for file in os.listdir(opt.annotation_path) if file.strip('.txt')[-1] ==str(split)])

    data = []                                     # (filename , lab_id)

    for file in split_lab_filenames:
        class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
        f = open(os.path.join(opt.annotation_path, file), 'r')
        for line in f:
            # If training data
            if train == 1 and line.split(' ')[1] == '1':
                frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                if opt.only_RGB and os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
                elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))

            # Elif validation/test data
            elif train != 1 and line.split(' ')[1] == '2':
                frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                if opt.only_RGB and os.path.exists(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))
                elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                    data.append((line.split(' ')[0][:-4], class_id))

        f.close()
    return data


if '__name__' == '__main__':
    opt = parse_opts()
    all_cls = get_cls(opt)

