"""
Copyright (c) 2017, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                   )




    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])


    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)



def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text)


def plot_convnet(model_info, font_size=8, to_file=None, flag_omit=True):
    fc_unit_size = 2

    patches = []
    colors = []

    font = {'size': font_size}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    
    layer_width = model_info["input_shape"][0] * 2
    # 1. Input image
    if flag_omit:
        add_layer_with_omission(patches, colors, size=model_info["input_shape"][:-1],
                                num=model_info["input_shape"][-1],
                                num_max=NumConvMax,
                                num_dots=NumDots,
                                top_left=[0, 0],
                                loc_diff=[3, -3])
    else:
        add_layer(patches, colors, 
                  size=model_info["input_shape"][:-1],
                  num=model_info["input_shape"][-1],
                  top_left=[0, 0], loc_diff=[3, -3])
    label([0, 0], "Inputs\n{}@{}x{}".format(model_info["input_shape"][2], 
                                            model_info["input_shape"][0],
                                            model_info["input_shape"][1]))
    
    # 2. Layers
    for idx, (layer_below_text, output_shape, patch_size) in enumerate(model_info["layers"]):
        if len(output_shape) == 3:
            size = output_shape[:-1]
        else:
            size = (fc_unit_size, fc_unit_size)
        n_maps = output_shape[-1]
        top_left = [(idx + 1) * layer_width, 0]
        
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size, num=n_maps,
                                    num_max=NumConvMax, num_dots=NumDots,
                                    top_left=top_left,
                                    loc_diff=[3, -3])
        else:
            add_layer(patches, colors, size=size, num=n_maps,
                      top_left=top_left,
                      loc_diff=[3, -3])
        if idx == len(model_info["layers"]) - 1:
            label_upper_text = "Outputs"
            if len(output_shape) == 3:
                label_upper_text += '\n{}@{}x{}'.format(
                    output_shape[2],
                    output_shape[0],
                    output_shape[1])
            else:
                label_upper_text += '\n{}'.format(output_shape[0])
        elif len(output_shape) == 3:
            label_upper_text = 'Feature\nmaps\n{}@{}x{}'.format(
                output_shape[2],
                output_shape[0],
                output_shape[1])
        else:
            label_upper_text ='Hidden\nunits\n{}'.format(output_shape[0])
            
        label(top_left, label_upper_text)
        
        # In-between layers
        start_ratio = [0.4, 0.5] if idx % 2 == 0 else [0.4, 0.8]
        end_ratio = start_ratio
        if layer_below_text.startswith("Convolution") or layer_below_text.startswith("Max-Pooling") or \
                layer_below_text.startswith("Avg-Pooling"):
            prev_shape = model_info["layers"][idx - 1][1] if idx > 0 else model_info["input_shape"]
            prev_size = prev_shape[:-1]
            if flag_omit:
                n_shown_prev = min(prev_shape[-1], NumConvMax)
                n_shown = min(output_shape[2], NumConvMax)
            add_mapping(patches, colors, start_ratio, end_ratio,
                        patch_size, 0,
                        [[top_left[0] - layer_width, 0], top_left], 
                        [[3, -3]] * 2, [n_shown_prev, n_shown], [prev_size, size])
        label(top_left, layer_below_text, 
                xy_off=[-layer_width // 2, - int(1.2 * layer_width)])

    ############################
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    plt.axis('equal')
    plt.axis('off')
    fig.set_size_inches(8, 2.5)
    
    if to_file is not None:
        fig.savefig(to_file, bbox_inches='tight', pad_inches=0)


def plot_keras_convnet(input_image, model, font_size=8, to_file=None, flag_omit=True):
    from .keras_model import read_model
    model_info = read_model(input_image, model)
    plot_convnet(model_info, font_size=font_size, to_file=to_file, flag_omit=flag_omit)


def plot_pytorch_convnet(input_image, model, font_size=8, to_file=None, flag_omit=True):
    from .pytorch_model import read_model
    model_info = read_model(input_image, model)
    plot_convnet(model_info, font_size=font_size, to_file=to_file, flag_omit=flag_omit)
