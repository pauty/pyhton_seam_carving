import sys
from skimage import io, color
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    if(rgb.ndim == 2):
        return rgb
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#sobelH = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype='double')
#sobelV = sobelH.T

#compute horizontal and vertical gradients
#gH = ndimage.filters.convolve(gray_img, sobelH, mode='constant', cval=0.0)
#gH = ndimage.filters.convolve(gray_img, sobelH, mode='constant', cval=0.0)

class SeamCarver:
    def __init__(self,img,protect=None,erase=None):
        self.original_img = img
        self.use_color = img.ndim == 3
        self.work_img = np.array(self.original_img, dtype='int')
        self.extend_img = True
        self.mark_const = 10000
        self.protect_mask = None
        self.erase_mask = None
        if(protect is not None):
            self.protect_mask = (protect >= 128).astype(int) * 1000
        if(erase is not None):
            self.erase_mask = (erase >= 128).astype(int) * -1000
    
    def image_retarget(self,w=0,h=0):
        if(w > 0):
            self.target_w = w
        else:
            self.target_w = self.work_img.shape[1]
        if(h > 0):
            self.target_h = h
        else:
            self.target_h = self.work_img.shape[0]
            
        #remove seams
        h_reached = self.work_img.shape[0] <= self.target_h
        w_reached = self.work_img.shape[1] <= self.target_w
        while(not h_reached or not w_reached):
            count = 0
            while(not w_reached and (count < 3 or h_reached)):
                self.compute_M()
                seam = self.find_seam()
                self.remove_seam(seam)
                count = count+1
                w_reached = self.work_img.shape[1] <= self.target_w
                print("reached size (%d,%d)" %(self.work_img.shape[1],self.work_img.shape[0]))
            count = 0
            self.work_img = self.work_img.T
            while(not h_reached and (count < 3 or w_reached)):
                self.compute_M()
                seam = self.find_seam()
                self.remove_seam(seam)
                count = count+1
                print("reached size (%d,%d)" %self.work_img.shape)
                h_reached = self.work_img.shape[1] <= self.target_h
            self.work_img = self.work_img.T
         
        #insert seams
        diff = self.target_w - self.work_img.shape[1]
        if(diff > 0):
            seamlist = []
            self.compute_M()
            for k in range(0,diff):
                seam = self.find_seam(mark=True)
                seamlist.append(seam)   
            seamlist = sorted(seamlist, key=lambda s: s[-1]) 
            c = 0
            for seam in seamlist:
                self.insert_seam(seam,c)
                c = c+1
                print("reached size (%d,%d)" %(self.work_img.shape[1],self.work_img.shape[0]))
                
        diff = self.target_h - self.work_img.shape[0]
        if(diff > 0):
            seamlist = []
            self.work_img = self.work_img.T
            self.compute_M()
            for k in range(0,diff):
                seam = self.find_seam(mark=True)
                seamlist.append(seam)   
            seamlist = sorted(seamlist, key=lambda s: s[-1])
            c = 0
            for seam in seamlist:
                self.insert_seam(seam,c)
                c = c+1
                print("reached size (%d,%d)" %self.work_img.shape)
            self.work_img = self.work_img.T
                     
    def get_pixel(self,i,j):
        if(self.extend_img):
            i = max(0,min(i,self.work_img.shape[0]-1))  #index in range
            j = max(0,min(j,self.work_img.shape[1]-1))
            return self.work_img[i,j]
        else:
            if(i < 0 or j < 0 or i >= self.work_img.shape[0] or j >= self.work_img.shape[1]):
                return 0
            return self.work_img[i,j]

    def get_cost_L(self,i,j):
        res = abs(self.get_pixel(i,j+1)-self.get_pixel(i,j-1)) + abs(self.get_pixel(i-1,j)-self.get_pixel(i,j-1))
        if(self.use_color):
            res = sum(res)
        return self.M[i-1,j-1] + res
        
    def get_cost_U(self,i,j):
        res = abs(self.get_pixel(i,j+1)-self.get_pixel(i,j-1)) 
        if(self.use_color):
            res = sum(res)
        return self.M[i-1,j] + res

    def get_cost_R(self,i,j):
        res = abs(self.get_pixel(i,j+1)-self.get_pixel(i,j-1)) + abs(self.get_pixel(i-1,j)-self.get_pixel(i,j+1))
        if(self.use_color):
            res = sum(res)
        return self.M[i-1,j+1] + res
        
    def compute_M(self):
        self.M = np.zeros((self.work_img.shape[0],self.work_img.shape[1]), dtype ='int')
        if(self.protect_mask is not None):
            self.M = self.M + self.protect_mask
        if(self.erase_mask is not None):
            self.M = self.M + self.erase_mask
        rows_M, coloumns_M = self.M.shape
        for i in range(1, rows_M):
            for j in range(1, coloumns_M-1):
                #note: you must add old value of M (not simply overwrite) to take in account masks
                self.M[i,j] = self.M[i,j] + min(self.get_cost_L(i,j), self.get_cost_U(i,j), self.get_cost_R(i,j))
            self.M[i,0] = self.M[i,0] + min(self.get_cost_U(i,0), self.get_cost_R(i,0))
            self.M[i,coloumns_M-1] = self.M[i,coloumns_M-1] + min(self.get_cost_L(i,coloumns_M-1), self.get_cost_U(i,coloumns_M-1))    

    def find_seam(self,mark=False):
        rows_M, coloumns_M = self.M.shape
        seam = np.zeros(shape=(rows_M,), dtype='int')
        minindex = np.argmin(self.M[rows_M-1])
        if(mark):
            self.M[rows_M-1,minindex] += self.mark_const
        seam[-1] = minindex
        for i in range(rows_M-2, -1, -1):
            mid = minindex
            if(mid != 0):
                if(self.M[i,mid-1] < self.M[i,minindex]):
                    minindex = mid-1
            if(mid != coloumns_M-1):
                if(self.M[i,mid+1] < self.M[i,minindex]):
                    minindex = mid+1
            seam[i] = minindex
            if(mark):
                self.M[i,minindex] += self.mark_const 
        return seam
    
    def remove_seam(self,seam):
        #print(seam)
        if(self.use_color):
            new_img = np.zeros(shape=(self.work_img.shape[0],self.work_img.shape[1]-1,self.work_img.shape[2]), dtype='int')
        else:
            new_img = np.zeros(shape=(self.work_img.shape[0],self.work_img.shape[1]-1), dtype='int')
        for i in range(0,seam.shape[0]):
            new_img[i] = np.delete(self.work_img[i], seam[i], axis=0)
        if(self.protect_mask is not None):
            new_pmask = np.zeros(shape=(self.work_img.shape[0],self.work_img.shape[1]-1), dtype='int')
            for i in range(0,seam.shape[0]):
                new_pmask[i] = np.delete(self.protect_mask[i], seam[i], axis=0)
            self.protect_mask = new_pmask            
        if(self.erase_mask is not None):
            new_emask = np.zeros(shape=(self.work_img.shape[0],self.work_img.shape[1]-1), dtype='int')
            for i in range(0,seam.shape[0]):
                new_emask[i] = np.delete(self.erase_mask[i], seam[i], axis=0)
            self.erase_mask = new_emask
        self.work_img = new_img
        
    
    #d is the number of seam already inserted: seams found in cost matrix must be shifted right by 1 for every past insertion
    def insert_seam(self,seam,d):
        if(self.use_color):
            new_img = np.zeros(shape=(self.work_img.shape[0],self.work_img.shape[1]+1,self.work_img.shape[2]), dtype='int')
        else:
            new_img = np.zeros(shape=(self.work_img.shape[0],self.work_img.shape[1]+1), dtype='int')
        for r in range(0,new_img.shape[0]):
            c = seam[r] + d
            new_img[r,:c] = self.work_img[r,:c]
            if(self.use_color):
                new_img[r,c] = ((self.get_pixel(r,c-1)+self.get_pixel(r,c))/2).round()
            else:
                new_img[r,c] = round((self.get_pixel(r,c-1)+self.get_pixel(r,c))/2)
            new_img[r,c+1:] = self.work_img[r,c:]
        self.work_img = new_img


#### MAIN #####
if len(sys.argv) < 2:
    print("Seam carving of image.\n\nUsage: %s filename.jpg [-p, -e]" % sys.argv[0])
    sys.exit(-1)

filepath = sys.argv[1]
try:
    image = misc.imread(filepath)
except FileNotFoundError:
    print("File " + filepath + " not found, exiting.")
    sys.exit(0)
    
filepath_list = filepath.split('/')
filename = filepath_list[-1]
p_mask = None
e_mask = None
if('-p' in sys.argv):
    mask_path = "/".join(filepath_list[:-1] + [filename.replace(".jpg","P.jpg")])
    try:      
        p_mask = io.imread(mask_path)
        if(p_mask.shape[0] != image.shape[0] or p_mask.shape[1] != image.shape[1]):
            print("protect mask size is different from image size and will be ignored.")
            p_mask = None
        else:
            p_mask = rgb2gray(p_mask)
    except FileNotFoundError:
        print("option -p specified but " + mask_path + " not found")
if('-e' in sys.argv):
    mask_path = "/".join(filepath_list[:-1] + [filename.replace(".jpg","E.jpg")])
    try:      
        e_mask = io.imread(mask_path)
        if(e_mask.shape[0] != image.shape[0] or e_mask.shape[1] != image.shape[1]):
            print("erase mask size is different from image size and will be ignored.")
            e_mask = None
        else:
            e_mask = rgb2gray(e_mask)
    except FileNotFoundError:
        print("option -e specified but " + mask_path + " not found")
  
plt.imshow(np.uint8(image))
plt.show()
lab = color.rgb2lab(image)
lab_scaled = (lab + [0, 128, 128]) / [100, 255, 255]
plt.imshow(lab_scaled)
plt.show()
sc = SeamCarver(image, p_mask, e_mask)
print("Loaded image size is %dx%d (WxH).\nPlease enter target size (non positive values mantain original size)" %(sc.original_img.shape[1],sc.original_img.shape[0]))
while True:
    try:
        w = int(input("Target width: "))
        h = int(input("Target height: "))
        break;
    except ValueError:
        print('Please enter valid values')
#pass target W and H
sc.image_retarget(w,h)
plt.imshow(np.uint8(sc.work_img), cmap='gray')
plt.show()
while True:
    try:
        choice = input("Do you want to save modified image? (y/n) ")
        choice = choice.upper()
        if(choice in ["Y","YES"]):
            save = True
            break
        elif(choice in ["N","NO"]):
            save = False
            break        
    except ValueError:
        pass
if(save):
    if(w <= 0):
        w = sc.original_img.shape[1]
    if(h <= 0):
        h = sc.original_img.shape[0]
    save_path = "/".join(filepath_list[:-1] + [filename.replace(".jpg",str(w) + "x" + str(h) + ".jpg")])
    misc.imsave(save_path, sc.work_img) 
    print("image successfully saved.") 

sys.exit(0)

