from time import sleep
import matlab.engine
# eng = matlab.engine.start_matlab()
# print(eng.matchScans)

if __name__ == "__main__":
    eng = matlab.engine.start_matlab()
    print("Matlab started")
    while (True):
        sleep(1)
    
