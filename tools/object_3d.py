import numpy as np

class bbox(object):
    ''' 3d object label '''
    def __init__(self, xmin, ymin, xmax, ymax):
        self.x_offset = xmin
        self.y_offset = ymin
        self.width = xmax-xmin
        self.height = ymax-ymin

class location(object):
    ''' 3d object label '''
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        # extract label, truncated, occluded
        self.kind_name = data[0]  # 'Car', 'Pedestrian', ...
        self.truncated = data[1]  # truncated pixel ratio [0..1]
        self.occluded = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]
        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.bbox = bbox(self.xmin, self.ymin, self.xmax, self.ymax) #np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        # extract 3d bounding box information
        self.height = data[8]  # box height
        self.width = data[9]  # box width
        self.length = data[10]  # box length (in meters)
        self.location = location(data[11], data[12], data[13])  # location (x,y,z)
        self.yaw = data[14]  # yaw angle
        if len(data) > 15:
            self.score = data[15]
    def print_object(self):
        print('kind_name, truncated, occluded, alpha: %s, %d, %d, %f' % \
              (self.kind_name, self.truncated, self.occluded, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox height,width,length: %f, %f, %f' % \
              (self.height, self.width, self.length))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.location.x, self.location.y, self.location.z, self.yaw))