'''
Python class to perform inference on individual images using specified DLC network (e.g. to be used on live camera feed).
Please see companion GUI for full program to record video while performing inference.

GK 12/05/2019

.. todo::

    Notes from Jonny:

    * I actually don't see dynamic cropping actually implemented anywhere, but i don't really understand the codebase all that well.
    * I think it's more general to ask for a method that returns a frame rather than an object that has a specific set of methods.

'''

import cv2
import os
from pathlib import Path
import numpy as np
from skimage.util import img_as_ubyte
import threading

# import tensorflow as tf

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config

class DLCLive(object):
    '''
    Parameters:
    -----------
    config : string
        Full path of the config.yaml file as a string.
    cropping : list of int
        cropping parameters in pixel number: [x1, x2, y1, y2]
    iteration : int, optional
        which iteration to use
    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.
    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).
    gputouse: int, optional
        Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    useFrozen: bool, optional
        use frozen tensorflow model (speeds up inference)
    TFGPUinference: bool, optional
        Perform inference on GPU with Tensorflow code. Introduced in "Pretraining boosts out-of-domain robustness for pose estimation" by
        Alexander Mathis, Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis Source: https://arxiv.org/abs/1909.11229
    dynamic_cropping: either False or tuple containing (detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal).
    processor: :class:`~.processor.Processor`
        User-defined processor object. Must contain two methods: process and save.
        The 'process' method takes in a pose, performs some processing, and returns processed pose.
        The 'save' method saves any valuable data created by or used by the processor
        Processors can be used for two main purposes:
        i) to run a forward predicting model that will predict the future pose from past history of poses (history can be stored in the processor object, but is not stored in this DLCLive object)
        ii) to trigger external hardware based on pose estimation (e.g. see 'TeensyLaser' processor)
    camera: callable
        method that returns a frame from a camera

    Attributes
    ----------------
    configured: bool
        whether the configuration file has been loaded and appropriately reconfigured based on input arguments
    cfg: dict
        (leaving this to u to document)
    dlc_cfg: dict
        (leaving this to u to document)
    model_folder: str
        path to model (idk how else you would document this)
    scorer: str
    scorerlegacy: str
    sess:
    inputs:
    outputs:

    '''

    def __init__(self, config, cropping=None, iteration=None, shuffle=1, trainingsetindex=0,
                 gputouse=None, useFrozen=True, TFGPUinference=False, dynamic_cropping=False,
                 processor=None, frame_method=None):

        self.cropping         = cropping
        self.iteration        = iteration
        self.shuffle          = shuffle
        self.trainingsetindex = trainingsetindex
        self.gputouse         = gputouse
        self.useFrozen        = useFrozen
        self.TFGPUinference   = TFGPUinference
        self.dynamic_cropping = dynamic_cropping
        self.processor        = processor
        self.frame_method     = frame_method
        self.poses            = []

        # load config
        # get absolute path in case a "~/" type path or relative path was given
        self.cfg_file     = os.path.abspath(config)
        self.cfg          = None
        self.dlc_cfg      = None
        self.model_folder = None
        self.scorer       = None
        self.scorerlegacy = None
        self.configured   = False
        self.init_config()

        # start tf prediction
        self.sess        = None
        self.inputs      = None
        self.outputs     = None
        self.pose_tensor = None
        self.init_prediction()

    def init_config(self):
        """
        Load the config.yaml and pose_config.yaml files and configure based on object arguments
        """

        if os.path.basename(self.cfg_file) == "config.yaml":
            # find the pose_config.yaml file
            self.cfg = auxiliaryfunctions.read_config(self.cfg_file)

            # warn if overriding project_path
            # override in case project was copied from somewhere else without changing config
            if os.path.dirname(self.cfg_file) != self.cfg['project_path']:
                Warning('config.yaml not in project_path, using the directory of config.yaml')
                self.cfg["project_path"] = os.path.dirname(self.cfg_file)

            # get model folder
            trainFraction = self.cfg['TrainingFraction'][self.trainingsetindex]
            self.model_folder = os.path.join(self.cfg["project_path"],
                                       str(auxiliaryfunctions.GetModelFolder(
                                           trainFraction, self.shuffle, self.cfg))

                                       )
            path_test_config = Path(self.model_folder) / 'test' / 'pose_cfg.yaml'
            try:
                self.dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle {} and trainFraction {} does not exist.".format(
                    self.shuffle, trainFraction)
                )

        else:
            ValueError('config given at instantiation must be the config.yaml file at the root of the project directory!')


        # gpu selection
        if self.gputouse:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gputouse)

        if self.iteration:
            self.cfg['iteration'] = self.iteration

        if self.cropping is not None:
            self.cfg['cropping']=True
            self.cfg['x1'], self.cfg['x2'], self.cfg['y1'], self.cfg['y2'] = self.cropping
            print("Overwriting cropping parameters:", self.cropping)

        # Check which snapshots are available and sort them by # iterations
        try:
            snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(self.model_folder , 'train'))if "index" in fn])
        except FileNotFoundError:
            raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle {} has not been trained/does not exist.".format(self.shuffle) +
                                    "Please train it before using it to analyze videos." +
                                    "Use the function 'train_network' to train the network for shuffle {}.".format(self.shuffle))

        if self.cfg['snapshotindex'] == 'all':
            Warning("Snapshotindex is set to 'all' in the config.yaml file." +
                    "Running video analysis with all snapshots is very costly!"+
                    "Use the function 'evaluate_network' to choose the best the snapshot." +
                    "For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex = self.cfg['snapshotindex']

        # sort snapshots by number of training (intervals??? what u call those?)
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in snapshots])
        snapshots = snapshots[increasing_indices]

        print("Using {} for model {}".format(snapshots[snapshotindex], self.model_folder))

        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check if data already was generated:
        self.dlc_cfg['init_weights'] = os.path.join(self.model_folder , 'train', snapshots[snapshotindex])
        trainingsiterations = (self.dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]

        # Update number of output and batchsize
        self.dlc_cfg['num_outputs'] = self.dlc_cfg.get('num_outputs', 1)
        self.dlc_cfg['batch_size'] = 1

        # (don't need to have a third value for state, if dynamic_cropping is anything that
        # evaluates True (ie. any non-None or non-zero tuple) then this will work
        # I also don't see this get setup anywhere else??
        # -jls 2020-03-09
        if self.dynamic_cropping:
            #(state,detectiontreshold,margin)=dynamic
            print("Starting analysis in dynamic cropping mode with parameters:", self.dynamic_cropping)
            self.dlc_cfg['num_outputs']=1
            self.TFGPUinference=False
            print("Switching num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode).")

        # Name for scorer:
        self.scorer, self.scorerlegacy = auxiliaryfunctions.GetScorerName(self.cfg, self.shuffle, trainFraction, trainingsiterations=trainingsiterations)
        if self.dlc_cfg['num_outputs']>1:
            if self.TFGPUinference:
                print("Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently.")
                self.TFGPUinference=False
            print("Extracting ", self.dlc_cfg['num_outputs'], "instances per bodypart")

        self.configured = True

    def init_prediction(self):
        """
        Initialize tensorflow session, inputs, and outputs depending on configuration
        """
        if not self.configured:
            RuntimeError('init_config has not been run!')

        if self.useFrozen:
            self.sess, self.inputs, self.outputs = predict.setup_frozen_prediction(self.dlc_cfg)
        elif self.TFGPUinference:
            self.sess, self.inputs, self.outputs = predict.setup_GPUpose_prediction(self.dlc_cfg)
            self.pose_tensor = predict.extract_GPUprediction(self.outputs, self.dlc_cfg)
        else:
            self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(self.dlc_cfg)

    def get_pose(self, frame=None):
        """

        Parameters
        ----------
        frame :class:`numpy.ndarray`
            image to extract pose. Should be an RGB, uint8 image, but if grayscale (len(frame.shape)== 2)
            or if given as float will be automatically converted.

        Returns
        -------

        """

        if frame is None:
            if self.frame_method is None:
                raise DLCLiveException("No frame provided for live pose estimation")
            else:
                frame = self.frame_method()

        # These conversions can be pretty costly. Usually I like EAFP but afaik tensorflow doesn't handle
        # it nicely, so let's
        #   a) allow frame conversion to be handled by whatever is giving them to us
        #   b) only do unambiguous operations - eg. grayscale to RGB, but how do we know if a (n,m,3) image is RGB or BGR?
        #   c)  check if we need to 8-bit-ify it before doing so
        # -jls 2020-03-09

        if self.cfg['cropping']:
            frame = frame[self.cfg['y1']:self.cfg['y2'],
                          self.cfg['x1']:self.cfg['x2']]

        # if we don't have a third axis, this is a grayscale image!!!
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if not frame.dtype == 'uint8':
            frame = img_as_ubyte(frame)

        # for each, expand dimensions to fit in with (batch_size, height, width, channels)
        # expand_dims returns a view, so it's v cheap.
        if self.useFrozen:
            pose = self.sess.run(self.outputs, feed_dict={
                self.inputs: np.expand_dims(frame, axis=0)
            })
        elif self.TFGPUinference:
            pose = self.sess.run(self.pose_tensor, feed_dict={
                self.inputs: np.expand_dims(frame, axis=0)
            })
            pose[:, [0,1,2]] = pose[:, [1,0,2]]
        else:
            pose = predict.getpose(frame, self.dlc_cfg, self.sess, self.inputs, self.outputs)

        # apply processor if any
        if self.processor:
            pose = self.processor.process(pose)

        return pose

    #
    # def _pose_on_thread(self):
    #     while self.continue_stream:
    #         if self.frame_method.new_frame:
    #             self.poses.append(self.get_pose())
    #
    #
    # def start_pose_stream(self):
    #     if self.frame_method is None:
    #         raise DLCException("DLCLive object does not have a camera. Cannot start pose stream without a camera.")
    #     self.continue_stream = True
    #     threading.Thread(target=self._pose_on_thread).start()
    #
    #
    # def stop_pose_stream(self):
    #     self.continue_stream = False


class DLCLiveException(Exception):
    ''' Raise when no frame is passed to DLCLive's get_pose '''
    pass
