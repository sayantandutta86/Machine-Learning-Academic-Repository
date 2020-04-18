#HW9: Human Action Recognition.
#Helper functions.
#Sayantan Dutta



import numpy as np
import cv2
import h5py


def getUCF101(base_directory=''):

    # action class labels
    class_file = open(base_directory + 'ucfTrainTestlist/classInd.txt', 'r')
    lines = class_file.readlines()
    lines = [line.split(' ')[1].strip() for line in lines]
    class_file.close()
    class_list = np.asarray(lines)

    # training data
    train_file = open(base_directory + 'ucfTrainTestlist/trainlist01.txt', 'r')
    lines = train_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0] for line in lines]
    y_train = [int(line.split(' ')[1].strip()) - 1 for line in lines]
    y_train = np.asarray(y_train)
    filenames = [base_directory + filename for filename in filenames]
    train_file.close()

    train = (np.asarray(filenames), y_train)

    # testing data
    test_file = open(base_directory + 'ucfTrainTestlist/testlist01.txt', 'r')
    lines = test_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0].strip() for line in lines]
    classnames = [filename.split('/')[1] for filename in filenames]
    y_test = [np.where(classname == class_list)[0][0]
              for classname in classnames]
    y_test = np.asarray(y_test)
    filenames = [base_directory + filename for filename in filenames]
    test_file.close()

    test = (np.asarray(filenames), y_test)

    return class_list, train, test

'''There is a text file provided with the dataset which lists all of the relative paths for the videos for the train/test split. This function can be called with the location of the dataset (/projects/training/bayw/hdf5/UCF-101-hdf5/) and returns three variables. class_list is a list of the action categories. train is a tuple. The first element is a numpy array with the absolute filepaths for the videos for training. The second element is a numpy array of class indices (0-100). test is a tuple in the same format as train but for the test dataset.'''


def loadFrame(args):
  
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)

    curr_w = 320
    curr_h = 240
    height = width = 224
    (filename, augment) = args

    data = np.zeros((3, height, width), dtype=np.float32)

    try:
        # load file from HDF5
        filename = filename.replace('.avi', '.hdf5')
        filename = filename.replace('UCF-101', 'UCF-101-hdf5')
        h = h5py.File(filename, 'r')
        nFrames = len(h['video']) - 1
        frame_index = np.random.randint(nFrames)
        frame = h['video'][frame_index]

        if augment:
            # RANDOM CROP - crop 70-100% of original size
            # don't maintain aspect ratio
            if np.random.randint(2) == 0:
                resize_factor_w = 0.3 * np.random.rand() + 0.7
                resize_factor_h = 0.3 * np.random.rand() + 0.7
                w1 = int(curr_w * resize_factor_w)
                h1 = int(curr_h * resize_factor_h)
                w = np.random.randint(curr_w - w1)
                h = np.random.randint(curr_h - h1)
                frame = frame[h:(h + h1), w:(w + w1)]

            # FLIP
            if np.random.randint(2) == 0:
                frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (width, height))
            frame = frame.astype(np.float32)

            # Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness + 1) - brightness / 2.0
            frame += random_add
            frame[frame > 255] = 255.0
            frame[frame < 0] = 0.0

        else:
            # don't augment
            frame = cv2.resize(frame, (width, height))
            frame = frame.astype(np.float32)

        # resnet model was trained on images with mean subtracted
        frame = frame / 255.0
        frame = (frame - mean) / std
        frame = frame.transpose(2, 0, 1)
        data[:, :, :] = frame

    except:
        print("Exception: " + filename)
        data = np.array([])

    return data


'''The above function is used for loading a single frame from a particular video in the dataset. args is a tuple with the first argument being the location of a video (which is in train[0] and test[0] from the getUCF101() function) and the second argument specifies whether data augmentation should be performed. The video paths in train and test look something like this: /projects/training/bayw/hdf5/UCF-101-hdf5/FloorGymnastics/v_FloorGymnastics_g23_c01.avi. However, since the dataset has now been saved as hdf5 files, the first part of this code converts this to /projects/training/bayw/hdf5/UCF-101/FloorGymnastics/v_FloorGymnastics_g23_c01.hdf5. A random frame is selected and if augment==True, the frame is randomly cropped, resized to the appropriate dimension for the model, flipped, and has its brightness adjusted. The frame is normalized based on the provided mean and std for the default PyTorch pretrained ResNet-50 model. Finally the data is returned and is a numpy array of size [3,224,224] of type np.float32. This is used for part 1.'''


def loadSequence(args):
    
    mean = np.asarray([0.433, 0.4045, 0.3776], np.float32)
    std = np.asarray([0.1519876, 0.14855877, 0.156976], np.float32)

    curr_w = 320
    curr_h = 240
    height = width = 224
    num_of_frames = 16

    (filename, augment) = args

    data = np.zeros((3, num_of_frames, height, width), dtype=np.float32)

    try:
        # load file from HDF5
        filename = filename.replace('.avi', '.hdf5')
        filename = filename.replace('UCF-101', 'UCF-101-hdf5')
        h = h5py.File(filename, 'r')
        nFrames = len(h['video']) - 1
        frame_index = np.random.randint(nFrames - num_of_frames)
        video = h['video'][frame_index:(frame_index + num_of_frames)]

        if augment:
            # RANDOM CROP - crop 70-100% of original size
            # don't maintain aspect ratio
            resize_factor_w = 0.3 * np.random.rand() + 0.7
            resize_factor_h = 0.3 * np.random.rand() + 0.7
            w1 = int(curr_w * resize_factor_w)
            h1 = int(curr_h * resize_factor_h)
            w = np.random.randint(curr_w - w1)
            h = np.random.randint(curr_h - h1)
            random_crop = np.random.randint(2)

            # Random Flip
            random_flip = np.random.randint(2)

            # Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness + 1) - brightness / 2.0

            data = []
            for frame in video:
                if random_crop:
                    frame = frame[h:(h + h1), w:(w + w1), :]
                if random_flip:
                    frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (width, height))
                frame = frame.astype(np.float32)

                frame += random_add
                frame[frame > 255] = 255.0
                frame[frame < 0] = 0.0

                frame = frame / 255.0
                frame = (frame - mean) / std
                data.append(frame)
            data = np.asarray(data)

        else:
            # don't augment
            data = []
            for frame in video:
                frame = cv2.resize(frame, (width, height))
                frame = frame.astype(np.float32)
                frame = frame / 255.0
                frame = (frame - mean) / std
                data.append(frame)
            data = np.asarray(data)

        data = data.transpose(3, 0, 1, 2)

    except:
        print("Exception: " + filename)
        data = np.array([])

    return data

'''This is very similar to the loadFrame() function. The 3D ResNet-50 model used for part 2 is trained on sequences of length 16. This function simply grabs a random subsequence of frames and augments them all in the exact same way (this is important when performing data augmentation on videos). This function returns a numpy array of size [3,16,224,224]. The last three channels must be the time and space dimensions since the PyTorch 3D convolution implementation acts on the last three channels of an input with size [batch_size,num_of_input_features,time,height,width].'''