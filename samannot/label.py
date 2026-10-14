import numpy as np
class PPoint:
    def __init__(self,x,y,pt_type,idx):
        self.x = x
        self.y = y
        self.pt_type = pt_type
        self.idx = idx
class PBox:
    def __init__(self,fx,fy,x,y,pt_type,idx):
        self.fx = fx
        self.fy = fy
        self.x = x
        self.y = y
        self.pt_type = pt_type
        self.idx = idx
class Label:
    def __init__(self, name, col=None):
        self.name = name
        self.col = col
        self.pts = {}
        self.boxes = {}
        self.features = []
        self.prop_frames = {}
    def find_most_recent_feature(self,idx, img_idx):
        most_recent_idx = -1
        feature_id = -1
        for i, modification in enumerate(self.features[idx]):
            if modification[1] <= img_idx:
                most_recent_idx = np.amax([most_recent_idx,modification[1]])
                feature_id = i
        if most_recent_idx != -1:
            return self.features[idx][feature_id][0]
        return " "
    def modify_feature(self, feature_idx, feature_value, frame_idx):
        self.features[feature_idx].append((feature_value, frame_idx))
    def add_pt(self, x,y, pt_type,idx,block_idx):
        if block_idx not in self.pts:
            self.pts[block_idx] = []
        self.pts[block_idx].append(PPoint(x,y,pt_type,idx))
        return len(self.pts)-1
    def add_box(self,fx, fy, x, y, pt_type,idx,block_idx):
        if block_idx not in self.boxes:
            self.boxes[block_idx] = []
        self.boxes[block_idx].append(PBox(fx,fy,x,y,pt_type,idx))
        return len(self.pts)-1
    def remove_pt(self, idx, block_idx):
        if 0 <= idx < len(self.pts[block_idx]):
            del self.pts[block_idx][idx]
    def remove_box(self, idx, block_idx):
        if 0 <= idx < len(self.boxes[block_idx]):
            del self.boxes[block_idx][idx]
    def clear_pts(self,block_idx):
        self.pts[block_idx] = []
    def to_dict(self):
        return { 'name': self.name, 'color': self.col,'pts': self.pts,'features': self.features,}
class Label_Handler:
    def __init__(self):
        self.colour_map = self.generate_color_map()
        self.current_label_idx = 0
    def generate_color_map(self,N=256): # PASCAL VOC
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap[i] = np.array([b, g, r])
        return cmap[1:]
    def convert_to_hexadecimal(self,colour):
        return f'#{colour[2]:02x}{colour[1]:02x}{colour[0]:02x}'
    def create_new_label(self,name):
        label_colour = self.convert_to_hexadecimal(self.colour_map[self.current_label_idx%len(self.colour_map)])
        self.current_label_idx += 1
        return Label(name,label_colour)