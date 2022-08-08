# Importing Libraries
import serial
import time

arduino = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=.1)
time.sleep(2)

def slowdown():
    arduino.write(b'L')

def stop():
    arduino.write(b'S')
    
def gas():
    arduino.write(b'F')
    
class Velocity(object):
    def __init__(self):
        self.convert = {10: slowdown(),
                        0: stop(),
                        15: gas()}

    def write_data(self, value):
        if arduino.isOpen():
            if value == 15:
                arduino.write(b'F')     
            elif value == 10:
                arduino.write(b'L')
            elif value == 5:
                arduino.write(b'S')
            # self.convert[int(value)]
            print ("")
            time.sleep(0.1)
        else:
            print ("Arduino Port Not Open !")
            arduino.close()

        if value is None:
            print ("No Value From Detector!")
            self.convert[0]
            arduino.close()
    
    def closed_connection():
        arduino.close()
        print ("GoodBye")



