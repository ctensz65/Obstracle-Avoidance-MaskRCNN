import serial
import time
import os
import sys

ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)


def main():
    while True:
        user_input = input("on / off / q : ")
        if user_input == "on":
            print("LED is on...")
            time.sleep(0.1)
            ser.write(b'F')
        elif user_input == "off":
            print("LED is off...")
            time.sleep(0.1)
            ser.write(b'L')
        elif user_input == "q":
            print("Quitting...")
            time.sleep(0.1)
            ser.write(b'S')
            ser.close()
            break
        else:
            print("Invalid input. Type on / off / quit.")
            ser.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        ser.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
