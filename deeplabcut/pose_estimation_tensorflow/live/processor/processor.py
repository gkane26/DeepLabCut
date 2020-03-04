"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

'''
Default processor class. Processors must contain two methods:
i) process: takes in a pose, performs operations, and returns a pose
ii) save: saves any internal data generated by the processor (such as timestamps for commands to external hardware)
'''

class Processor(object):

    def __init__(self):
        pass

    def process(self, pose):
        return pose

    def save(self, file=''):
        return 0
