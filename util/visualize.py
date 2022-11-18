import os
import io

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model.utils import OffsetGenerator


def vis_attention(model, basepath, side=14, q=(7, 7)):

    attentions_model = []
    for idx, layer in enumerate(model.blocks):
        phi_p = layer.attn.get_weight(model.get_rpe())[0]
        OffsetGenerator.initialize(side, pad_size=0)
        offset = OffsetGenerator.get_qk_vec().norm(p=2, dim=-1).unsqueeze(0)
        nonlocality = (offset * phi_p).mean(dim=[-1, -2])
        nonlocality, sort_idx = nonlocality.sort()

        # sort attentions in the order of nonlocality measure
        weights = phi_p[:, q[0]*side+q[1]].view(-1, side, side).cpu().detach()[sort_idx]

        attentions = []
        for hdx, w in enumerate(weights):
            savepath = '%s/weight/L%d_H%d.png' % (basepath, idx, hdx)
            save_attention(w, savepath, q)
            attentions.append(Image.open(savepath))
        attentions_model.append(vertical_stack(attentions, gap=20))
    horizontal_stack(attentions_model, gap=20).save('%s/weight_%s.png' % (basepath, os.path.basename(basepath)))


def vertical_stack(pil_imgs, gap):
    canvas_height = sum([pil.size[1] for pil in pil_imgs])
    canvas_height += ((len(pil_imgs) - 1) * gap)
    canvas_width = max([pil.size[0] for pil in pil_imgs])
    canvas = Image.new('RGB', (canvas_width, canvas_height), "WHITE")

    ypos = 0
    for pil in pil_imgs:
        canvas.paste(pil, (0, ypos))
        ypos += (pil.size[1] + gap)

    return canvas


def horizontal_stack(pil_imgs, gap):
    canvas_width = sum([pil.size[0] for pil in pil_imgs])
    canvas_width += ((len(pil_imgs) - 1) * gap)
    canvas_height = max([pil.size[1] for pil in pil_imgs])
    canvas = Image.new('RGB', (canvas_width, canvas_height), "WHITE")

    xpos = 0
    for pil in pil_imgs:
        canvas.paste(pil, (xpos, 0))
        xpos += (pil.size[0] + gap)

    return canvas

def save_attention(attn, savepath, q):
    rect = patches.Rectangle((q[1]-0.5, q[0]-0.5), 1, 1, linewidth=4, edgecolor='r', facecolor='none')
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.set_axis_off()

    if (attn - attn).sum() == 0:
        caxes = axes.matshow(attn, interpolation ='nearest', vmin=0.0, vmax=1.0)
    else:
        caxes = axes.matshow(attn, interpolation ='nearest')

    axes.add_patch(rect)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    plt.close()

