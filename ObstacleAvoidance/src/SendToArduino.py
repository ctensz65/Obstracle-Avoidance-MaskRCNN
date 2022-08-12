# Importing Libraries
# <kiri, 140>
#
#

import serial
import time

arduino = serial.Serial('COM5', baudrate=115200, timeout=.1)
time.sleep(2)

startMarker = 60
endMarker = 62


def sendToArduino(sendStr):
    arduino.write(sendStr.encode('utf-8'))  # change for Python3


def recvFromArduino():
    global startMarker, endMarker

    ck = ""
    x = "z"  # any value that is not an end- or startMarker
    byteCount = -1  # to allow for the fact that the last increment will be one too many

    # wait for the start character
    while ord(x) != startMarker:
        x = arduino.read()

    # save data until the end marker is found
    while ord(x) != endMarker:
        if ord(x) != startMarker:
            ck = ck + x.decode("utf-8")  # change for Python3
            byteCount += 1
        x = arduino.read()

    return(ck)


def write_data(value):
    waitingForReply = False
    if arduino.isOpen():
        if waitingForReply == False:
            sendToArduino(value)
            print("\nSent from PC " + value)
            waitingForReply = True

        if waitingForReply == True:
            while arduino.inWaiting() == 0:
                pass

            dataRecvd = recvFromArduino()
            print("Reply Received " + dataRecvd)
            waitingForReply = False

            print("===========")

        time.sleep(0.1)
    else:
        print("Arduino Port Not Open !")
        arduino.close()


def closed_connection():
    msgStop = "<stop, 0>"
    sendToArduino(msgStop)
    arduino.close()
    print("GoodBye")
