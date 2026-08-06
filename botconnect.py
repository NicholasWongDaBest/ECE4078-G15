import socket
import struct
import threading
import time
import numpy as np
import cv2


class BotConnect:
    '''
    This class handles the communication between the pibot/server and your pc
    PID constants are sent to the robot at the start
    Camera images and wheel movement are continuously being transferred between the robot and your pc
    '''
    def __init__(self, robot_ip, wheel_port=8000, camera_port=8001, pid_config_port=8002):
        self.robot_ip = robot_ip
        self.wheel_port = wheel_port
        self.camera_port = camera_port
        self.pid_config_port = pid_config_port
        
        # Connection state
        self.running = True
        
        # Robot state
        self.left_speed = 0
        self.right_speed = 0
        self.left_count = 0
        self.right_count = 0
        self.move_mode = 0 # 0 is manual driving, 1 is autonomous based on time, 2 is autonomous based on encoder count
        self.speed_lock = threading.Lock()
        self.autonomous_done = False
        
        # Camera frame
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # Start connection threads
        self.command_thread = threading.Thread(target=self._connect_wheel)
        self.command_thread.daemon = True
        self.command_thread.start()
        self.camera_thread = threading.Thread(target=self._connect_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
    
    def set_pid(self, use_pid, kp, ki, kd):
        """Send PID constants to the robot"""
        try:
            # Open a temporary socket for PID configuration
            pid_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            pid_socket.connect((self.robot_ip, self.pid_config_port))
            
            # Pack PID constants and send
            pid_data = struct.pack("!ffff", use_pid, kp, ki, kd)
            pid_socket.sendall(pid_data)
            
            # Wait for acknowledgment
            response = pid_socket.recv(4)
            status = struct.unpack("!i", response)[0]
            
            pid_socket.close()
            return status == 1
        except Exception as e:
            print(f"Failed to set PID: {str(e)}")
            return False      
    
    def move_manual(self, wheel_speed):
        # Change the robot speed here. The value should be between -1 and 1.
        # Note that this is just a number specifying how fast the robot should go, not the actual speed in m/s
        with self.speed_lock:
            self.left_speed = max(min(wheel_speed[0], 1), -1)
            self.right_speed = max(min(wheel_speed[1], 1), -1)
            self.move_mode = 0
        
    def move_auto_time(self, wheel_speed, duration):
        # call this function to move the robot autonomously for a specified duration (seconds)
        # to determine if the movement is done, check the self.autonomous_done flag
        with self.speed_lock:
            self.left_speed = max(min(wheel_speed[0], 1), -1)
            self.right_speed = max(min(wheel_speed[1], 1), -1)
            self.duration = duration
            self.autonomous_done = False
            self.move_mode = 1
        
    def move_auto_encoder(self, wheel_speed, target_left_enc, target_right_enc):
        # call this function to move the robot autonomously for a specified number of encoder counts
        with self.speed_lock:
            self.left_speed = max(min(wheel_speed[0], 1), -1)
            self.right_speed = max(min(wheel_speed[1], 1), -1)
            self.target_left_enc = target_left_enc
            self.target_right_enc = target_right_enc
            self.autonomous_done = False
            self.move_mode = 2
    
    def get_image(self):
        with self.frame_lock: # need to lock when multiple threads access the same data, especially if data is bigger in size
            return self.frame.copy() if self.frame is not None else np.zeros([360,480,3], dtype=np.uint8)
    
    def get_encoder_counts(self):
        return self.left_count, self.right_count
    
    def stop(self):
        self.move_manual([0,0])
        time.sleep(0.2)
        # self.running = False
    
    def _connect_wheel(self):
        # Thread function to handle wheel connection (sending speed and receiving encoder counts)
        while self.running:
            try:
                wheel_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                wheel_socket.connect((self.robot_ip, self.wheel_port))
                print("Connected to wheel server")
                prev_speed_manual = []
                while self.running:
                    try:
                        with self.speed_lock:
                            mode = self.move_mode
                            l_speed = self.left_speed
                            r_speed = self.right_speed
                            duration = getattr(self, 'duration', None)
                            t_left_enc = getattr(self, 'target_left_enc', None)
                            t_right_enc = getattr(self, 'target_right_enc', None)
                        # Manual driving mode
                        if mode == 0:
                            # if prev_speed_manual != (l_speed, r_speed): # only send if speeds are new
                            #     data_send = struct.pack("!Bff", mode, l_speed, r_speed)
                            #     wheel_socket.sendall(data_send)
                                
                            #     # Receive encoder counts
                            #     # .recv is a blocking call and waits indefinitely until data is received
                            #     data_recv = wheel_socket.recv(8)
                            #     if not data_recv or len(data_recv) != 8:
                            #         print("Wheel server disconnected")
                            #         break
                                
                            #     # Update encoder counts
                            #     self.left_count, self.right_count = struct.unpack("!ii", data_recv)
                            #     prev_speed_manual = (l_speed, r_speed)


                            data_send = struct.pack("!Bff", mode, l_speed, r_speed)
                            wheel_socket.sendall(data_send)
                            
                            # Receive encoder counts
                            data_recv = wheel_socket.recv(8)
                            if not data_recv or len(data_recv) != 8:
                                print("Wheel server disconnected")
                                break
                            
                            # Update encoder counts
                            self.left_count, self.right_count = struct.unpack("!ii", data_recv)
                            prev_speed_manual = (l_speed, r_speed)  # can keep for reference, no longer gates anything
                    
                        # Autonomous mode, based on time. Time is monitored on the robot/server side.
                        if mode == 1:
                            data_send = struct.pack("!Bfff", mode, l_speed, r_speed, duration)
                            wheel_socket.sendall(data_send)
                            
                            # Receive encoder counts
                            # .recv is a blocking call and waits indefinitely until data is received
                            data_recv = wheel_socket.recv(8)
                            if not data_recv or len(data_recv) != 8:
                                print("Wheel server disconnected")
                                break
                            
                            # Update encoder counts
                            self.left_count, self.right_count = struct.unpack("!ii", data_recv)
                            self.autonomous_done = True
                            self.left_speed, self.right_speed = 0, 0
                            self.move_mode = 0
                        
                        # Autonomous mode, based on encoder count. Count is monitored on the robot/server side.
                        if mode == 2:
                            data_send = struct.pack("!Bffii", mode, l_speed, r_speed, t_left_enc, t_left_enc)
                            wheel_socket.sendall(data_send)
                            
                            # Receive encoder counts
                            # .recv is a blocking call and waits indefinitely until data is received
                            data_recv = wheel_socket.recv(8)
                            if not data_recv or len(data_recv) != 8:
                                print("Wheel server disconnected")
                                break
                            
                            # Update encoder counts
                            self.left_count, self.right_count = struct.unpack("!ii", data_recv)
                            self.autonomous_done = True
                            self.left_speed, self.right_speed = 0, 0
                            self.move_mode = 0
                            
                        time.sleep(0.01)
                        
                    except Exception as e:
                        print(f"Wheel error: {str(e)}")
                        break
                    
                wheel_socket.close()
                
            except Exception as e:
                print(f"Unable to connect to wheel server: {str(e)}")
                time.sleep(1)  # Wait before trying to reconnect
    
    def _connect_camera(self):
        # Thread function to handle camera connection
        while self.running:
            try:
                camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                camera_socket.connect((self.robot_ip, self.camera_port))
                print("Connected to camera server")
                
                while self.running:
                    try:
                        # Signal ready for a new frame
                        camera_socket.sendall(b'\x01')
                        
                        size_data = camera_socket.recv(4, socket.MSG_WAITALL)
                        if not size_data or len(size_data) != 4:
                            print("Camera server disconnected")
                            break
                        
                        # Unpack data size & receive the JPEG data
                        jpeg_size = struct.unpack("!I", size_data)[0]
                        jpeg_data = camera_socket.recv(jpeg_size, socket.MSG_WAITALL)
                        if not jpeg_data or len(jpeg_data) != jpeg_size:
                            print("Incomplete frame received")
                            break
                        
                        image = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        with self.frame_lock: self.frame = image # Update the global frame with thread safety
                            
                    except Exception as e:
                        print(f"Camera error: {str(e)}")
                        break
                    
                camera_socket.close()
                
            except Exception as e:
                print(f"Unable to connect to camera server: {str(e)}")
                time.sleep(1)
                with self.frame_lock: self.frame = None # Reset frame when disconnected