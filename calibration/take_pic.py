import os
import sys
import cv2
import pygame
import argparse
import numpy as np
sys.path.insert(0, "..")
from botconnect import BotConnect # access the robot communication


class calibration:
    def __init__(self,args):
        self.botconnect = BotConnect(args.ip)
        self.img = np.zeros([360,480,3], dtype=np.uint8)
        self.command = {'wheel_speed':[0, 0], 'image': False}
        self.image_collected = 0
        self.finish = False

    def image_collection(self, dataDir):
        if self.command['image']:
            image = self.botconnect.get_image()
            filename = os.path.join(dataDir, 'calib_' + str(self.image_collected) + '.png')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image)
            self.image_collected +=1
            print('Collected {} images for camera calibration.'.format(self.image_collected))               
        if self.image_collected == images_to_collect:
                self.finish = True
        self.command['image']= False

    def update_keyboard(self):
        for event in pygame.event.get():
            # Replace with your M1 codes if you want to drive around and take picture
            # Holding the Pibot to take pictures are also fine
            
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                pass # TODO:
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                pass # TODO:
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                pass # TODO:
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                pass # TODO:
            elif event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.command['wheel_speed'] = [0, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.command['image'] = True
            
            self.botconnect.move_manual(self.command['wheel_speed'])

    def take_pic(self):
        self.img = self.botconnect.get_image()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    args, _ = parser.parse_known_args()

    currentDir = os.getcwd()
    dataDir = os.path.join(currentDir,'calib_pics')
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    
    images_to_collect = 20 # feel free to change this

    calib = calibration(args)

    width, height = 640, 480
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Calibration')
    canvas.fill((0, 0, 0))
    pygame.display.update()
    
    # collect data
    print('Collecting {} images for camera calibration.'.format(images_to_collect))
    print('Press ENTER to capture image.')
    while not calib.finish:
        calib.update_keyboard()
        calib.take_pic()
        calib.image_collection(dataDir)
        img_surface = pygame.surfarray.make_surface(calib.img)
        img_surface = pygame.transform.flip(img_surface, True, False)
        img_surface = pygame.transform.rotozoom(img_surface, 90, 1)
        canvas.blit(img_surface, (0, 0))
        pygame.display.update()
    print('Finished image collection.\n')
    print('Images Saved at: \n',dataDir)

