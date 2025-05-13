import cv2
import numpy as np
from datetime import datetime, timedelta
import imageio
import copy
from matplotlib import colors

from kiui.op import make_divisible

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='00:00', help="start time, should follow --format")
    parser.add_argument('--end', type=str, default='01:00', help="end time (inclusive), should follow --format")
    parser.add_argument('--format', type=str, default='%M:%S', help="time format, follow datetime convention (https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)")
    parser.add_argument('--delta', type=int, default=1, help="delta in seconds")
    parser.add_argument('--out', type=str, default='timer.mp4', help="output file name")
    parser.add_argument('--speedup', type=int, default=1, help="speed up time")
    parser.add_argument('--pad', type=int, default=10, help="video padding pixels")
    parser.add_argument('--fontscale', type=int, default=5, help="font scale (also video scale)")
    parser.add_argument('--thickness', type=int, default=8, help="font thickness")
    parser.add_argument('--fg_color', type=str, default='black', help="text color, see https://matplotlib.org/stable/gallery/color/named_colors.html")
    parser.add_argument('--bg_color', type=str, default='white', help="background color, see https://matplotlib.org/stable/gallery/color/named_colors.html")

    opt = parser.parse_args()

    delta = timedelta(seconds=opt.delta)
    start_time = datetime.strptime(opt.start, opt.format)
    end_time = datetime.strptime(opt.end, opt.format)
    countdown = start_time > end_time
    if countdown:
        end_time -= delta
    else:
        end_time += delta

    font = cv2.FONT_HERSHEY_SIMPLEX
    lineType = cv2.LINE_AA

    fontColor = [int(x * 255) for x in colors.to_rgb(opt.fg_color)]
    bgColor = [int(x * 255) for x in colors.to_rgb(opt.bg_color)]

    text_width, text_height = cv2.getTextSize(start_time.strftime(opt.format), font, opt.fontscale, lineType)[0]

    # padding
    W = make_divisible(text_width + opt.pad * 2)
    H = make_divisible(text_height + opt.pad * 2)
    location = (opt.pad, text_height + opt.pad)

    print(f'[INFO] video size: {H} x {W}')

    images = []

    cur_time = copy.deepcopy(start_time)

    while True:
        image = np.full((H, W, 3), fill_value=bgColor, dtype=np.uint8)
        text = cur_time.strftime(opt.format)
        # print(text, image.shape, cur_time)
        cv2.putText(image, text, location, font, opt.fontscale, fontColor, opt.thickness, lineType)
        images.append(image)
        if countdown:
            cur_time -= delta
        else:
            cur_time += delta
        
        if cur_time == end_time:
            break

    images = np.stack(images, axis=0)
    imageio.mimwrite(opt.out, images, fps=opt.speedup, quality=8, macro_block_size=1)