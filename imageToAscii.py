import numpy as np
import math

import PIL
from PIL import Image
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw

import argparse
import sys
import cv2

import os

GRAY_SCALE_MAP = {
    1 : "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    2 : "*+=-:. ",
    3 : "86547  "
}

def averagePixelColor(image):

    imArr = np.array(image)

    w,h = imArr.shape

    return np.average(imArr.reshape(w*h))


def imageToAscii(fileName, cols, scale, style):

    image = Image.open(fileName).convert('L')

    W, H = image.size

    w = W / cols
    h = w / scale

    rows = int(H/h)

    if (cols > W or rows > H):
        print "Image is too small"  
        exit(0)

    imageRows = []

    for i in range(rows):
        y1 = int(i * h )
        y2 = int ((i + 1) * h)

        if i == rows - 1:
            y2 = H

        imageRows.append("")

        for j in range(cols):
            x1 = int(j * w)
            x2 = int((j + 1) * w)

            if j == cols - 1:
                x2 = W

            img = image.crop((x1,y1,x2,y2))

            avg = int(averagePixelColor(img))

            value = style[int((avg * len(style) - 1 ) / 255)]
            print "Appending a new value to str row "
            imageRows[i] +=  value

    return imageRows

def main_program(args):
    if not args.image:
        print "Please enter an iamge to convert!"
        exit(0)

    if args.style not in range(1,4):
        print  "Not an available style option"
        exit(0)

    imageRows = imageToAscii(args.image, args.cols, args.scale, GRAY_SCALE_MAP[args.style])

    f = open(args.out, "w")

    for r in imageRows:
        f.write(r + '\n')
    f.close()


def splitAndConvertFrames():
    file_name = "test.mp4"
    file_folder = "asciiFrames"
    cap = cv2.VideoCapture(file_name)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        file_name = 'frame' + str(i) + '.jpg'

        if ret == False:
            break

        #Creates a new file_name
        cv2.imwrite(file_name, frame)

        # When the write is complete then convert that video frame into ascii

        # Save the generated ascii frames into the asciiFrames directory

        imageRows = imageToAscii(file_name, 120, 0.63, GRAY_SCALE_MAP[3])

        asciiFileName = 'frame' + str(i) + '.txt'

        f = open(os.path.join(file_folder,asciiFileName), "w")

        for r in imageRows:
            f.write(r + '\n')
        f.close()
        i += 1
        
    cap.release()
    cv2.destroyAllWindows()

def concatImageFrames():
    image_folder = "asciiFrames"
    video_name = "out.mp4"

    images = [img for img in os.listdir(image_folder) if img.endswith(".txt")]
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    j = 0
    for image in images:

        grayscale = 'L'

        if image == 'frame30.txt':
            image = 'frame29.txt'
        text_file = open(image, "r")
        lines =  tuple(l.rstrip() for l in text_file.readlines())



        large_font = 20
        font_path = 'cour.ttf'

        try:
            font = PIL.ImageFont.truetype(font_path, size=large_font)
        except IOError:
            font = PIL.ImageFont.load_default()
            print "Font not available"

        pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
        max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
        # max height is adjusted down because it's too large visually for spacing
        test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        max_height = pt2px(font.getsize(test_string)[1])
        max_width = pt2px(font.getsize(max_width_line)[0])
        height = max_height * len(lines)  # perfect or a little oversized
        width = int(round(max_width + 40))  # a little oversized
        image = PIL.Image.new(grayscale, (width, height), color=PIXEL_OFF)
        draw = PIL.ImageDraw.Draw(image)

        vertical_position = 5
        horizontal_position = 5
        line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
        for line in lines:
            draw.text((horizontal_position, vertical_position),
                    line, fill=PIXEL_ON, font=font)
            vertical_position += line_spacing
        # crop the text
        c_box = PIL.ImageOps.invert(image).getbbox()
        image = image.crop(c_box)

        cv2.imwrite(os.path.join(image_folder, 'img' + str(i) + '.png'))

        j += 1

        #video.write(cv2.imread(os.path.join(image_folder, image)))



    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    
    cap.release()
    cv2.destroyAllWindows()



# Setting up the args parser
parser = argparse.ArgumentParser(description="Set env variables here")
parser.add_argument('--cols', default=120, type=int)
parser.add_argument('--scale', default=0.63, type=float)
parser.add_argument('--style', default=1, type=int)
parser.add_argument('--image', type=str)
parser.add_argument('--out', type=str, default="drawing.txt")
parser.add_argument('--testing', type=int, default=1)







# This is so you can retrieve the variables that were set in the command line
args = parser.parse_args()

if args.testing == 1:
    #splitAndConvertFrames()
    concatImageFrames()
else:
    main_program(args)







