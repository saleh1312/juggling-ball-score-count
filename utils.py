import cv2
import numpy as np
import math

class utils:
    def segment(self,img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([100,250,204])
        upper_range = np.array([105,255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        kernel = np.ones((5, 5), 'uint8')

        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return x+(0.5*w),y+(0.5*h),w,h
    
    def calc_speed(self,img1,img2):
        x1,y1,w1,h1=self.segment(img1)
        x2,y2,w2,h2=self.segment(img2)
        speedx= x2-x1
        speedy=y2-y1
        return speedx,speedy
    
    def calc_velocty(self,sx1,sy1,sx2,sy2):
        vx=(sx2-sx1)/2
        vy=(sy2-sy1)/2
        return vx,vy
    def sign(self,value):
        if value >0:
            return 1
        elif value <0:
            return -1
        return 0
        
        
        
        
        
#x,y,w,h=ut.segment(img)
    

#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
          
if __name__ =="__main__":
    
    cap=cv2.VideoCapture('ball.wmv')
    ut = utils()
    first_frame=True
    
    sx1=0
    sy1=0
    frame1=None
    while True:
        if first_frame==True:
            
            ret1,img1 = cap.read()
            ret2,img2 = cap.read()
            sx,sy=ut.calc_speed(img1, img2)
            
            sx1=sx
            sy1=sy
            frame1=img2
            
            first_frame=False
            
        else:
            ret2,img2 = cap.read()
            if ret2==False:
                break
            sx,sy=ut.calc_speed(frame1, img2)
            if sy ==0 and sx==0:
                continue
                
            vx,vy=ut.calc_velocty(sx1, sy1, sx, sy)
            if (ut.sign(sy1) ==1 and ut.sign(sy) ==-1) :
                
                change_vel = math.sqrt(sy1*sy1 + sy*sy);
             
                if change_vel > 15:
                    print('bounce')
                    
            elif (ut.sign(sx1) != ut.sign(sx)):

                change_vel = math.sqrt(sx1*sx1 + sx*sx);
                if change_vel > 15:
                    print('bounce')
                
   
                
            
            sx1=sx
            sy1=sy
            frame1=img2
            
            
        cv2.imshow('img',img2)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
        
    cv2.destroyAllWindows()
            
            
        