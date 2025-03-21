"""
Functionality: Extract the Radiometric DataFrom Flir Images/Videos

"https://flir.custhelp.com/app/answers/detail/a_id/3504/~/flir-science-file-sdk-for-python---getting-started"

"""

import numpy as np
import os
import matplotlib.pyplot as plt
import fnv
import fnv.reduce
import fnv.file
from tkinter import filedialog
import cv2
import matplotlib.animation as animation


class DataExtraction:
    def __init__(self, filePath: str):

        self.im = fnv.file.ImagerFile(filePath)
        if self.im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
            # set units to temperature,
            self.im.unit = fnv.Unit.TEMPERATURE_FACTORY
            self.im.temp_type = fnv.TempType.CELSIUS  # temperature unit
        else:
            # if file has no temperature calibration,
            self.im.unit = fnv.Unit.COUNTS

        self.max_x = 0
        self.max_y = 0
        self.maxT = 0
        self.min_x = 0
        self.min_y = 0
        self.minT = 0

    def min_and_max(self, data):

        # find location of max in the whole image

        self.max_x, self.max_y = np.unravel_index(np.argmax(data), data.shape)
        self.maxT = data[(self.max_x, self.max_y)]  # find value of the maximum

        # repeat for minimum temp
        self.min_x, self.min_y = np.unravel_index(np.argmin(data), data.shape)
        self.minT = data[(self.min_x, self.min_y)]

    def data_extraction(self, save=False):

        video = []

        for i in range(self.im.num_frames):
            self.im.get_frame(i)

            data = np.array(self.im.final).reshape(self.im.height, self.im.width)

            video.append(data)

        video = np.transpose(np.stack(video), (1, 2, 0))  # Frame H W -->  H W Frame

        print(video.shape)

        if save:

            np.savez("seq43.npz", skin=video[:, :, 2:2000])

        else:
            return video

    def plot_data(self):

        frames = []
        fig = plt.figure()

        # Create an empty image to add the color bar
        first_frame = True

        for i in range(self.im.num_frames):
            self.im.get_frame(i)

            data = np.array(self.im.final, copy=False).reshape(
                self.im.height, self.im.width
            )

            img = plt.imshow(data, cmap="gist_rainbow", animated=True)

            # Add color bar only for the first frame
            if first_frame:
                cbar = plt.colorbar(img, ax=plt.gca())
                cbar.set_label("Intensity")
                first_frame = False

            frames.append([img])

        ani = animation.ArtistAnimation(
            fig, frames, interval=50, blit=True, repeat_delay=1000
        )

        plt.show()


def plotSinglePixel(video, x, y):
    npz = np.load(video)

    skin_lesion = npz["skin"]

    skin_lesion = skin_lesion[:, :, :200]

    print(skin_lesion.shape)

    data = []
    for i in range(skin_lesion.shape[2]):

        temperature = skin_lesion[:, :, i][x, y]

        # plt.imshow(skin_lesion[:,:,i])
        # plt.plot(x,y,'ro')
        # plt.show()

        data.append(temperature)

    plt.scatter(range(len(data)), data)
    plt.show()

    # print(data)
    # print(len(data))


# 355,215

if __name__ == "__main__":
    video_path = (
        "/home/nipun/Documents/Uni_Malta/Alive/Alive/DataExtraction/Rec-000043.seq"
    )
    dataExt = DataExtraction(video_path)
    # dataExt.plot_data()
    # dataExt.data_extraction(save=True)

    # Saved Data
    saved_video = "seq43.npz"

    plotSinglePixel(saved_video, 355, 215)
