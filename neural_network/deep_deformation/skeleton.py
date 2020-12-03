"""
@author: Vincent Bonnet
@description : Skeleton object to deal with hierarchy and skeleton text file
skeleton = Skeleton();
skeleton.load(filename)
skeleton.print_root()
"""

class Bone:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

class Skeleton:
    def __init__(self):
        self.root = None
        self.bones = {} # map name with bone

    def load(self, filename):
        with open(filename, 'r')  as file:
            # collect and add bones
            bone_data = []
            for line in file:
                if len(line)==0:
                    continue
                data = line.strip().split(',')
                bone_data.append(data)
                self.add_bone(bone_name=data[0])

            # parent bones
            for data in bone_data:
                self.parent(bone_name=data[0], parent_name=data[1])

    def add_bone(self, bone_name):
        if bone_name in self.bones:
            # already inserted
            return
        self.bones[bone_name] = Bone(bone_name)

    def parent(self, bone_name, parent_name):
        bone = self.bones[bone_name]
        parent = self.bones.get(parent_name, None)

        # parent None is considered the root
        if not parent:
            if not self.root:
                parent = Bone(parent_name)
                self.root = parent
            elif self.root.name != parent_name:
                raise Exception('multiple roots found')
            parent = self.root

        # set hierarchy
        parent.children.append(bone)
        bone.parent = parent

    def print_root(self):
        self._print_hierarchy(self.root)

    def _print_hierarchy(self, node=None, space=0):
        if not node:
            return

        print('|'+'-' * space + node.name)
        for child in node.children:
            self._print_hierarchy(child, space+2)


