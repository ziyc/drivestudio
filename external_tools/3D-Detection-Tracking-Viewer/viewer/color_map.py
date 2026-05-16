import matplotlib.pyplot as plt
import numpy as np


def generate_objects_color_map(color_map_name='rainbow'):
    """
    generate a list of random colors based on the specified color map name.
    reference  https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :param color_map_name: (str), the name of objects color map, such as "rainbow", "viridis","brg","gnuplot","hsv"
    :return: (list), a list of random colors
    """
    color_map = []
    np.random.seed(4)

    x = 0
    for i in range(10000):
        if x > 1:
            x = np.random.random() * 0.5
        color_map.append(x)
        x += 0.2
    cmp = plt.get_cmap(color_map_name)
    color_map = cmp(color_map)
    color_map = color_map[:, 0:3] * 255
    color_map = color_map.astype(int).tolist()
    return color_map

def generate_objects_colors(object_ids,color_map_list):
    """
    map the object indices into colors
    :param object_ids: (array or list(N,)), object indices
    :param color_map_list: (list(K,3)), color map list
    :return: (list(N,3)), a list of colors
    """
    assert len(color_map_list)>len(object_ids), "the color map list must longer than object indices list !"

    if len(object_ids)==0:
        return []
    else:
        colors=[]
        for i in object_ids:
            colors.append(color_map_list[i])
        return colors

def generate_scatter_colors(scatters,color_map_name='rainbow'):
    """
    map the scatters to colors
    :param scatters: (array or list(N,)),
    :param color_map_name: (str), the name of objects color map, such as "rainbow", "viridis","brg","gnuplot","hsv"
                             reference  https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :return: (array(N,4)), each item represents (red, green, blue, alpha),
    """
    if len(scatters)==0:
        return []
    else:
        scatters = np.array(scatters)
        scatters_max = scatters.max()
        scatters_min = scatters.min()

        div = scatters_max-scatters_min
        if div !=0:
            scatters = (scatters-scatters_min)/div
        cmp = plt.get_cmap(color_map_name)
        new_colors = cmp(scatters)
        new_colors = new_colors[:, 0:3] * 255
        alpha = np.ones(shape=(len(new_colors), 1)) * 255
        new_colors = np.concatenate([new_colors, alpha], -1)
    return new_colors.astype(int)


if __name__ == '__main__':
    a = range(0,10)
    b= range(0,10)
    co =range(0,10)

    colormap = generate_objects_color_map()
    colors = generate_objects_colors(co,colormap)
    print(colors)

    scatters = np.arange(0,10,0.1)

    colors = generate_scatter_colors(scatters)
    print(colors)
