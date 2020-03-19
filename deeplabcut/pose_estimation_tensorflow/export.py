"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import numpy as np
import ruamel.yaml
import glob
import shutil
import tarfile
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.nnet import predict


def create_deploy_config_template():
    '''

    CURRENTLY NOT IMPLEMENTED

    Creates a template for config.yaml file.
    This specific order is preserved while saving as yaml file.
    '''

    yaml_str = '''\
# Deploy config.yaml - info about project origin:
    Task:
    scorer:
    date:
    \n
# Project path
    project_path:
    \n
# Annotation data set configuration (and individual video cropping parameters)
    video_sets:
    bodyparts:
    \n
# Plotting configuration
    skeleton:
    skeleton_color:
    \n
    '''

    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile


def write_deploy_config(configname, cfg):
    '''

    CURRENTLY NOT IMPLEMENTED

    Write structured config file.
    '''

    with open(configname, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file, ruamelFile = create_deploy_config_template()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if not 'skeleton' in cfg.keys():
            cfg_file['skeleton'] = []
            cfg_file['skeleton_color'] = 'black'
        ruamelFile.dump(cfg_file, cf)


def load_model(cfg, shuffle=1, trainingsetindex=0, TFGPUinference=True):
    '''

    Loads a tensorflow session with a DLC model from the associated configuration

    return a tensorflow session with DLC model given cfg and shuffle

    Parameters:
    -----------
    cfg : dict
        Configuration read from the project's cfg.yaml file

    shuffle : int, optional
        which shuffle to use

    trainingsetindex : int. optional
        which training fraction to use, identified by its index

    TFGPUinference : bool, optional
        use tensorflow inference model? default = True

    Returns:
    --------
    sess : tensorflow session
        tensorflow session with DLC model from the provided configuration, shuffle, and trainingsetindex

    checkpoint file path : string
        the path to the checkpoint file associated with the loaded model
    '''

    ########################
    ### find snapshot to use
    ########################

    train_fraction = cfg['TrainingFraction'][trainingsetindex]
    model_folder = os.path.join(cfg['project_path'], str(auxiliaryfunctions.GetModelFolder(train_fraction, shuffle, cfg)))
    path_test_config = os.path.normpath(model_folder + '/test/pose_cfg.yaml')

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle, train_fraction))

    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(model_folder , 'train'))if "index" in fn])
    except FileNotFoundError:
      raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

    if cfg['snapshotindex'] == 'all':
        print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    ####################################
    ### Load and setup CNN part detector
    ####################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(model_folder, 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
    dlc_cfg['num_outputs'] = cfg.get('num_outputs', dlc_cfg.get('num_outputs', 1))
    #dlc_cfg['batch_size'] = cfg['batch_size']
    dlc_cfg['batch_size'] = None

    # load network
    if TFGPUinference:
        sess, _, _ = predict.setup_GPUpose_prediction(dlc_cfg)
        output = ['concat_1']
    else:
        sess, _, _ = predict.setup_pose_prediction(dlc_cfg)
        if dlc_cfg['location_refinement']:
            output = ['Sigmoid', 'pose/locref_pred/block4/BiasAdd']
        else:
            output = ['Sigmoid', 'pose/part_pred/block4/BiasAdd']

    input = tf.get_default_graph().get_operations()[0].name

    return sess, input, output, dlc_cfg


def tf_to_pb(sess, checkpoint, output, output_dir=None):
    '''

    Saves a frozen tensorflow graph (a protobuf file).
    See https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

    Parameters
    ----------
    sess : tensorflow session
        session with graph to be saved

    checkpoint : string
        checkpoint of tensorflow model to be converted to protobuf (output will be <checkpoint>.pb)

    output : list of strings
        list of the names of output nodes (is returned by load_models)

    output_dir : string, optional
        path to the directory that exported models should be saved to.
        If None, will export to the directory of the checkpoint file.
    '''

    output_dir = os.path.expanduser(output_dir) if output_dir else os.path.dirname(checkpoint)
    ckpt_base = os.path.basename(checkpoint)

    # save graph to pbtxt file
    pbtxt_file = os.path.normpath(output_dir + '/' + ckpt_base + '.pbtxt')
    tf.train.write_graph(sess.graph.as_graph_def(), '', pbtxt_file, as_text=True)

    # create frozen graph from pbtxt file
    pb_file = os.path.normpath(output_dir + '/' + ckpt_base + '.pb')

    freeze_graph.freeze_graph(input_graph=pbtxt_file,
                              input_saver='',
                              input_binary=False,
                              input_checkpoint=checkpoint,
                              output_node_names=",".join(output),
                              restore_op_name="save/restore_all",
                              filename_tensor_name="save/Const:0",
                              output_graph=pb_file,
                              clear_devices=True,
                              initializer_nodes='')


def export_model(cfg_path, iteration=None, shuffle=1, trainingsetindex=0, TFGPUinference=True, overwrite=False, make_tar=True):
    '''

    Export DLC models for the model zoo or for live inference.
    Saves the pose configuration, snapshot files, and frozen graph of the model to
        a directory named exported-models within the project directory

    Parameters
    -----------

    cfg_path : string
        path to the DLC Project config.yaml file

    iteration : int, optional
        the model iteration you wish to export.
        If None, uses the iteration listed in the config file

    shuffle : int, optional
        the shuffle of the model to export. default = 1

    trainingsetindex : int, optional
        the index of the training fraction for the model you wish to export. default = 1

    TFGPUinference : bool, optional
        use the tensorflow inference model? Default = True
        For inference using DeepLabCut-live, it is recommended to set TFGPIinference=False

    overwrite : bool, optional
        if the model you wish to export has already been exported, whether to overwrite. default = False

    make_tar : bool, optional
        Do you want to compress the exported directory to a tar file? Default = True
        This is necessary to export to the model zoo, but not for live inference.
    '''

    ### read config file

    try:
        cfg = auxiliaryfunctions.read_config(cfg_path)
    except FileNotFoundError:
        FileNotFoundError("The config.yaml file at %s does not exist." % cfg_path)

    cfg['project_path'] = os.path.dirname(cfg_path)
    cfg['iteration'] = iteration if iteration is not None else cfg['iteration']
    cfg['batch_size'] = cfg['batch_size'] if cfg['batch_size'] > 1 else 2

    ### load model

    sess, input, output, dlc_cfg = load_model(cfg, shuffle, trainingsetindex, TFGPUinference)
    ckpt = dlc_cfg['init_weights']
    model_dir = os.path.dirname(ckpt)

    ### set up export directory

    export_dir = os.path.normpath(cfg['project_path'] + '/' + 'exported-models')
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    sub_dir_name = 'DLC_%s_%s' % (cfg['Task'], dlc_cfg['net_type'])
    full_export_dir = os.path.normpath(export_dir + '/' + sub_dir_name)

    if os.path.isdir(full_export_dir):
        if not overwrite:
            raise FileExistsError("Export directory %s already exists. Terminating export..." % full_export_dir)
    else:
        os.mkdir(full_export_dir)

    ### write pose config file

    # sort dlc_cfg keys alphabetically, then save to pose_cfg.yaml in export directory
    dlc_cfg = dict(dlc_cfg)
    sorted_cfg = {}
    for key, value in sorted(dlc_cfg.items()):
        sorted_cfg[key] = value

    pose_cfg_file = os.path.normpath(full_export_dir + '/pose_cfg.yaml')
    ruamel_file = ruamel.yaml.YAML()
    ruamel_file.dump(sorted_cfg, open(pose_cfg_file, 'w'))

    ### copy checkpoint to export directory

    ckpt_files = glob.glob(ckpt+'*')
    ckpt_dest = [os.path.normpath(full_export_dir+'/'+os.path.basename(ckf)) for ckf in ckpt_files]
    for ckf, ckd in zip(ckpt_files, ckpt_dest):
        shutil.copy(ckf, ckd)

    ### create pbtxt and pb files for checkpoint in export directory

    tf_to_pb(sess, ckpt, output, output_dir=full_export_dir)

    ### tar export directory

    if make_tar:
        tar_name = os.path.normpath(full_export_dir + '.tar.gz')
        with tarfile.open(tar_name, 'w:gz') as tar:
            tar.add(full_export_dir, arcname=os.path.basename(full_export_dir))
