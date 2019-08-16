import cv2 as cv
import sqlite3
import sys
import os
import threading
import csv
import math

NUM_REGIONS = 2000
IOU_FACE_THRESHOLD = 0.5
IOU_NON_FACE_THRESHOLD = 0.35


def computeIOU(a, b):
    [xa, ya, wa, ha] = a
    [xb, yb, wb, hb] = b
    width = max(0, min(xa + wa, xb + wb) - max(xa, xb))
    height = max(0, min(ya + ha, yb + hb) - max(ya, yb))
    si = width * height
    sa = wa * ha
    sb = wb * wb
    su = sa + sb - si
    return si / su


def cropImage(img, rect):
    croppedImg = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    height = croppedImg.shape[0]
    width = croppedImg.shape[0]
    if width < 1 or height < 1:
        self.print("WARNING: cropped image invalid.")
        self.print("Image shape: {}")
        self.print("Rect: {}".format(rect))
    return croppedImg


class EntryWriter():

    def __init__(self, file):
        self.csvWriter = csv.writer(file)
        self.lock = threading.Lock()

    def write(self, filepath, region, isFace):
        self.lock.acquire()
        x, y, w, h = region
        self.csvWriter.writerow([filepath, x, y, w, h, isFace])
        self.lock.release()


class PreProcessor(threading.Thread):

    def __init__(self, dbFilename, pathToImageFolder, length, offset,
                 entryWriter, threadNumber):
        super(PreProcessor, self).__init__()
        self.dbFilename = dbFilename
        self.pathToImageFolder = pathToImageFolder
        self.length = length
        self.offset = offset
        self.ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.entryWriter = entryWriter
        self.outputFilePath = 'thread_{}'.format(threadNumber)

    def readImage(self, filepath):
        if not os.path.exists(filepath):
          self.print("Could not find image {}".format(filepath))
          return None
        return cv.imread(filepath)

    def getRegions(self, img):
        self.ss.setBaseImage(img)
        self.ss.switchToSelectiveSearchFast()
        # for some reason selective search fails for images with large width compared to height
        # we will just ignore these as there aren't many of them
        try:
          return self.ss.process()[:NUM_REGIONS]
        except:
          return ()

    def print(self, string):
        with open(self.outputFilePath, 'a') as f:
            f.write(string + '\n')

    def run(self):
        self.db = sqlite3.connect(dbFilename)
        for [fileId, filepath] in self.db.execute(
                'SELECT file_id,filepath FROM FaceImages LIMIT {} OFFSET {}'
                .format(self.length, self.offset)):

            self.print("processing {}".format(fileId))
            img = self.readImage(os.path.join(self.pathToImageFolder, filepath))
            if img is None:
              continue

            faceRects = self.db.execute(
                'SELECT x,y,w,h FROM Faces NATURAL JOIN FaceRect WHERE file_id = "{}"'
                .format(fileId)).fetchall()

            for region in self.getRegions(img):
                iouMax = 0
                regionIOUMax = [0, 0, 0, 0]
                for rect in faceRects:
                    iou = computeIOU(region, rect)
                    if iou > IOU_FACE_THRESHOLD:
                        self.entryWriter.write(filepath, region, True)
                        break
                    elif iou >= iouMax:
                        iouMax = iou
                        regionIOUMax = region
                else:
                    if iouMax < IOU_NON_FACE_THRESHOLD:
                        self.entryWriter.write(filepath, regionIOUMax, False)


def makeDirSafe(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    _, numThreadsString, dbFilename, pathToImageFolder = sys.argv
    db = sqlite3.connect(dbFilename)
    length, = db.execute("SELECT COUNT() FROM FaceImages").fetchone()
    numThreads = int(numThreadsString)
    batchSize = math.ceil(length / numThreads)
    with open('regions2.csv', 'w') as out:
        ew = EntryWriter(out)
        threads = []
        for i in range(numThreads):
            threads.append(
                PreProcessor(dbFilename, pathToImageFolder, batchSize,
                             i * batchSize, ew, i))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print("Done!")

